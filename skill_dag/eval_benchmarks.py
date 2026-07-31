"""
OLMES-style benchmark pass over the finished main runs. RUN ON GPU (inference only).

Why this exists (PLAN.md review item 16): the preregistered DV is held-out loss on the
same nine Dolma domains the models trained on. The source doc's weak-evidence claim is
about benchmarks -- "exactly the same benchmark scores with slightly reduced compute, or a
few pp higher on benchmark scores with equal compute". Held-out loss on the training
domains cannot speak to that, because it never leaves the training distribution. This adds
the external measurement, and it costs almost nothing: no gradients, one forward pass per
answer option, against 490 GPU-h of training.

This is a SECONDARY, DESCRIPTIVE measure. The preregistered decision rule is the loss
analysis in analyze.py; nothing here feeds it. Reporting a benchmark table is not licence
to switch DV after seeing results.

Scoring follows OLMES: rank the answer options by the model's length-normalised log
likelihood of each continuation given the context. Four metrics are reported per task
because at this scale accuracy alone is uninformative --

  acc                    argmax of total continuation logprob
  acc_per_char           argmax of logprob / continuation length in characters (primary)
  correct_prob           softmax over options of the total logprobs, mass on the gold option
  correct_prob_per_char  the same over length-normalised scores

-- and DataDecide's finding is the reason: at small scale, accuracy on these tasks sits at
chance and moves only in noise, while continuous likelihood metrics still separate models.
A 1B model trained on 2B tokens from a step-1000 checkpoint WILL be at or near chance on
most of these. The chance rate is reported alongside every task so that nobody reads a
2-point accuracy gap as a result. If `correct_prob_per_char` also fails to separate arms,
the honest conclusion is that this scale cannot resolve benchmark differences, which is
itself worth reporting against the doc's claim.

NETWORK: Farmshare compute nodes generally cannot reach the internet. Fetch the datasets
on a login node first with --download-only; they land in the shared HF cache and the GPU
job then runs offline.

Usage:
  python eval_benchmarks.py --download-only                     # login node, no GPU
  python eval_benchmarks.py --runs "runs/arm*" --out bench.json # after Phase 3
  python eval_benchmarks.py --runs runs/smoke --limit 200       # quick shakedown
  python eval_benchmarks.py --base-ref --out bench_base.json    # untrained reference
"""
import argparse, glob, json, math, os, statistics, time

import torch

from train_mixture import BASE_MODEL, BASE_REVISION

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# tasks
#
# Each formatter returns (pairs, gold) where pairs is [(context, continuation), ...],
# one per answer option. Per-option contexts are allowed because Winogrande varies the
# context rather than the continuation.
# ---------------------------------------------------------------------------

def _mc(question, options, gold):
    return [(f"Question: {question}\nAnswer:", " " + o) for o in options], gold


def _labelled(ex, stem_key):
    """ARC / OpenBookQA / CommonsenseQA all use a {label, text} choices struct."""
    ch = ex["choices"]
    labels, texts = list(ch["label"]), list(ch["text"])
    key = ex.get("answerKey")
    if key not in labels:
        return None
    return _mc(ex[stem_key], texts, labels.index(key))


def fmt_arc(ex):
    return _labelled(ex, "question")


def fmt_openbookqa(ex):
    return _labelled(ex, "question_stem")


def fmt_csqa(ex):
    return _labelled(ex, "question")


def fmt_hellaswag(ex):
    label = ex.get("label")
    if label in (None, ""):
        return None
    ctx = f"{ex['activity_label']}: {ex['ctx_a']} {ex['ctx_b'].capitalize()}".strip()
    return [(ctx, " " + e.strip()) for e in ex["endings"]], int(label)


def fmt_piqa(ex):
    if int(ex["label"]) < 0:
        return None
    return _mc(ex["goal"], [ex["sol1"], ex["sol2"]], int(ex["label"]))


def fmt_socialiqa(ex):
    opts = [ex["answerA"], ex["answerB"], ex["answerC"]]
    ctx = f"{ex['context']}\nQuestion: {ex['question']}\nAnswer:"
    return [(ctx, " " + o) for o in opts], int(ex["label"]) - 1


def fmt_boolq(ex):
    ctx = f"{ex['passage']}\nQuestion: {ex['question']}?\nAnswer:"
    return [(ctx, " no"), (ctx, " yes")], int(bool(ex["answer"]))


def fmt_winogrande(ex):
    # The blank is filled by the option, so the CONTEXT differs per option and the scored
    # continuation is the shared suffix. This is the standard partial-evaluation setup;
    # it also means acc and acc_per_char coincide here, since the continuations are equal.
    s = ex["sentence"]
    if "_" not in s:
        return None
    i = s.index("_")
    suffix = s[i + 1:]
    return [(s[:i] + ex["option1"], suffix), (s[:i] + ex["option2"], suffix)], \
        int(ex["answer"]) - 1


# name -> (hf path, config, split, formatter)
TASKS = {
    "arc_easy":      ("allenai/ai2_arc", "ARC-Easy", "test", fmt_arc),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test", fmt_arc),
    "boolq":         ("google/boolq", None, "validation", fmt_boolq),
    "csqa":          ("tau/commonsense_qa", None, "validation", fmt_csqa),
    "hellaswag":     ("Rowan/hellaswag", None, "validation", fmt_hellaswag),
    "openbookqa":    ("allenai/openbookqa", "main", "test", fmt_openbookqa),
    "piqa":          ("ybisk/piqa", None, "validation", fmt_piqa),
    "socialiqa":     ("allenai/social_i_qa", None, "validation", fmt_socialiqa),
    "winogrande":    ("allenai/winogrande", "winogrande_xl", "validation", fmt_winogrande),
}
# MMLU is deliberately absent: OLMES scores it few-shot, and a 5-shot prompt at this scale
# measures in-context-learning ability the models do not have yet. Adding it would produce
# a column of chance values that invites over-reading. Declared, not silently dropped.


def load_task(name, limit=None):
    from datasets import load_dataset
    path, cfg, split, fmt = TASKS[name]
    ds = load_dataset(path, cfg, split=split) if cfg else load_dataset(path, split=split)
    items = []
    for ex in ds:
        got = fmt(ex)
        if got is None:
            continue
        pairs, gold = got
        if not (0 <= gold < len(pairs)):
            continue
        items.append({"pairs": pairs, "gold": gold})
        if limit and len(items) >= limit:
            break
    return items


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_pairs(model, tok, pairs, device, max_len, batch_size):
    """-> [(sum_logprob, n_cont_tokens, n_cont_chars)] for each (context, continuation)."""
    encoded = []
    for ctx, cont in pairs:
        c_ids = tok(ctx, add_special_tokens=False)["input_ids"]
        k_ids = tok(cont, add_special_tokens=False)["input_ids"]
        if not k_ids:                      # empty continuation cannot be scored
            k_ids = [tok.eos_token_id]
        ids = (c_ids + k_ids)[-max_len:]
        # truncation must never eat the continuation, or the score is of a different string
        n_cont = min(len(k_ids), len(ids) - 1)
        encoded.append((ids, max(n_cont, 1), len(cont)))

    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    out = []
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i:i + batch_size]
        width = max(len(e[0]) for e in chunk)
        inp = torch.full((len(chunk), width), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), width), dtype=torch.long)
        for r, (ids, _, _) in enumerate(chunk):
            inp[r, :len(ids)] = torch.tensor(ids)
            att[r, :len(ids)] = 1
        inp, att = inp.to(device), att.to(device)
        logits = model(input_ids=inp, attention_mask=att).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)
        for r, (ids, n_cont, n_chars) in enumerate(chunk):
            total = 0.0
            # token at position t is predicted by the logits at t-1
            for t in range(len(ids) - n_cont, len(ids)):
                total += logprobs[r, t - 1, ids[t]].item()
            out.append((total, n_cont, n_chars))
    return out


def eval_task(model, tok, items, device, max_len, batch_size):
    n_correct = n_correct_pc = 0
    probs, probs_pc, gold_lp, chance = [], [], [], []
    for it in items:
        scored = score_pairs(model, tok, it["pairs"], device, max_len, batch_size)
        raw = [s for s, _, _ in scored]
        per_char = [s / max(c, 1) for s, _, c in scored]
        g = it["gold"]
        n_correct += int(max(range(len(raw)), key=lambda k: raw[k]) == g)
        n_correct_pc += int(max(range(len(per_char)), key=lambda k: per_char[k]) == g)
        probs.append(_softmax_at(raw, g))
        probs_pc.append(_softmax_at(per_char, g))
        gold_lp.append(raw[g])
        chance.append(1.0 / len(raw))
    n = len(items)
    return {
        "n": n,
        "acc": n_correct / n,
        "acc_per_char": n_correct_pc / n,
        "correct_prob": sum(probs) / n,
        "correct_prob_per_char": sum(probs_pc) / n,
        "mean_gold_logprob": sum(gold_lp) / n,
        "chance": sum(chance) / n,
    }


def _softmax_at(scores, idx):
    m = max(scores)
    ex = [math.exp(s - m) for s in scores]
    return ex[idx] / sum(ex)


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

def model_dir(run):
    """train_mixture.py writes the finished model to <run>/final; ckpt_resume is the
    periodic checkpoint and can lag the end of training by --ckpt-every-tokens."""
    for sub in ("final", "ckpt_resume"):
        p = os.path.join(run, sub)
        if os.path.exists(os.path.join(p, "config.json")):
            return p, sub
    return None, None


def load_model(path, revision, device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, revision=revision,
                                             torch_dtype=torch.bfloat16).to(device)
    m.config.use_cache = False
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None, help="run directories (globs ok)")
    ap.add_argument("--tasks", nargs="+", default=sorted(TASKS),
                    choices=sorted(TASKS), metavar="TASK")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap items per task; use for shakedown runs, not for results")
    ap.add_argument("--out", default="bench_summary.json")
    ap.add_argument("--base-ref", action="store_true",
                    help="also score the untrained base revision, as a floor to compare "
                         "the trained runs against")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--revision", default=BASE_REVISION)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="re-score runs that have bench.json")
    ap.add_argument("--download-only", action="store_true",
                    help="fetch the datasets into the HF cache and exit; run this on a "
                         "login node because compute nodes are usually offline")
    args = ap.parse_args()

    if args.download_only:
        for t in args.tasks:
            n = len(load_task(t, limit=None))
            print(f"  cached {t}: {n} items")
        print("\nDatasets cached. The GPU job can now run offline.")
        return

    if not args.runs and not args.base_ref:
        raise SystemExit("nothing to do: pass --runs and/or --base-ref")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, revision=args.revision)

    print("loading tasks")
    tasks = {}
    for t in args.tasks:
        tasks[t] = load_task(t, args.limit)
        print(f"  {t:14s} {len(tasks[t]):6d} items")
    if args.limit:
        print(f"  NOTE: --limit {args.limit} is set; these numbers are a shakedown, "
              f"not results")

    targets = []
    if args.base_ref:
        targets.append(("base", args.base, args.revision))
    for pat in args.runs or []:
        for d in sorted(glob.glob(pat)) or [pat]:
            path, which = model_dir(d)
            if not path:
                print(f"  skipping {d}: no final/ or ckpt_resume/ model")
                continue
            if which == "ckpt_resume":
                print(f"  NOTE: {d} has no final/; scoring {which}, which may lag the "
                      f"end of training")
            targets.append((d, path, None))

    summary = {}
    for name, path, rev in targets:
        dest = os.path.join(name, "bench.json") if os.path.isdir(name) else None
        if dest and os.path.exists(dest) and not args.force:
            summary[name] = json.load(open(dest))["tasks"]
            print(f"\n{name}: already scored, reusing bench.json (--force to redo)")
            continue
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        model = load_model(path, rev, args.device)
        res = {}
        for t in args.tasks:
            res[t] = eval_task(model, tok, tasks[t], args.device, args.max_len,
                               args.batch_size)
            r = res[t]
            print(f"  {t:14s} acc {r['acc']:.3f}  acc/char {r['acc_per_char']:.3f}  "
                  f"P(gold)/char {r['correct_prob_per_char']:.3f}  "
                  f"(chance {r['chance']:.3f})", flush=True)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
        blob = {"model": path, "tasks": res, "limit": args.limit,
                "seconds": round(time.time() - t0, 1)}
        if dest:
            json.dump(blob, open(dest, "w"), indent=2)
        summary[name] = res

    # ---- aggregate by arm, descriptively ----
    by_arm = {}
    for name, res in summary.items():
        arm = "base" if name == "base" else os.path.basename(name).rsplit("_seed", 1)[0]
        by_arm.setdefault(arm, []).append(res)

    print("\n=== by arm (mean over seeds) ===")
    agg = {}
    for arm, seeds in sorted(by_arm.items()):
        agg[arm] = {"n_seeds": len(seeds), "tasks": {}}
        for t in args.tasks:
            vals = {m: [s[t][m] for s in seeds]
                    for m in ("acc", "acc_per_char", "correct_prob_per_char")}
            agg[arm]["tasks"][t] = {
                m: {"mean": statistics.mean(v),
                    "sd": statistics.stdev(v) if len(v) > 1 else None}
                for m, v in vals.items()}
        macro = statistics.mean(
            agg[arm]["tasks"][t]["acc_per_char"]["mean"] for t in args.tasks)
        macro_p = statistics.mean(
            agg[arm]["tasks"][t]["correct_prob_per_char"]["mean"] for t in args.tasks)
        agg[arm]["macro_acc_per_char"] = macro
        agg[arm]["macro_correct_prob_per_char"] = macro_p
        print(f"  {arm:20s} n={len(seeds)}  macro acc/char {macro:.4f}  "
              f"macro P(gold)/char {macro_p:.4f}")

    json.dump({
        "per_model": summary, "by_arm": agg,
        "tasks": args.tasks, "limit": args.limit,
        "notes": [
            "SECONDARY AND DESCRIPTIVE. The preregistered decision rule is the loss "
            "analysis in analyze.py; nothing here feeds it.",
            "acc_per_char is the OLMES primary; compare every value against the task's "
            "chance rate before reading anything into it.",
            "at 1B params and 2B tokens accuracy is expected at or near chance, which is "
            "why the continuous correct_prob metrics are reported alongside",
            "MMLU is excluded: OLMES scores it few-shot and these models have no "
            "in-context-learning ability to measure",
        ],
    }, open(args.out, "w"), indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
