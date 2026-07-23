"""
Per-checkpoint mastery eval for the skill-DAG experiment. RUN ON GPU (cheap).

Walks every ckpt_tokens* checkpoint in a run directory and measures per-skill
held-out accuracy (greedy decode, exact match) -> one eval_log.jsonl per run,
consumed by analyze.py to build tokens-to-mastery curves.

Usage:
  python eval_mastery.py --run runs/topo_101 [--n-per-skill 300] [--batch-size 64]
"""
import argparse, json, os, re
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "skill_dag_dataset")


def load_heldout(n_per_skill):
    by_skill = defaultdict(list)
    with open(os.path.join(DATA, "heldout.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if len(by_skill[r["skill"]]) < n_per_skill:
                by_skill[r["skill"]].append(r)
    return by_skill


def eval_checkpoint(path, by_skill, batch_size, max_new_tokens):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    out = {}
    per_item = {}
    for skill, recs in sorted(by_skill.items()):
        hits = []
        for i in range(0, len(recs), batch_size):
            batch = recs[i : i + batch_size]
            enc = tok([r["prompt"] for r in batch], return_tensors="pt",
                      padding=True, padding_side="left").to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, g in zip(batch, texts):
                pred = g.strip().split("\n")[0].strip()
                ok = pred.startswith(r["answer"]) and (
                    len(pred) == len(r["answer"]) or not pred[len(r["answer"])].isdigit())
                hits.append(int(ok))
        out[skill] = sum(hits) / len(hits)
        per_item[skill] = hits  # kept for item-level bootstrap in analyze.py
    del model
    torch.cuda.empty_cache()
    return out, per_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing ckpt_tokens* dirs")
    ap.add_argument("--n-per-skill", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    args = ap.parse_args()

    by_skill = load_heldout(args.n_per_skill)
    ckpts = []
    for d in os.listdir(args.run):
        m = re.match(r"ckpt_tokens(\d+)", d)
        if m:
            ckpts.append((int(m.group(1)), os.path.join(args.run, d)))
    ckpts.sort()
    assert ckpts, f"no checkpoints found in {args.run}"

    log_path = os.path.join(args.run, "eval_log.jsonl")
    done_tokens = set()
    if os.path.exists(log_path):  # resume-safe
        with open(log_path) as f:
            done_tokens = {json.loads(l)["tokens"] for l in f}

    for tokens, path in ckpts:
        if tokens in done_tokens:
            print(f"skip {tokens:,} (already evaluated)")
            continue
        acc, per_item = eval_checkpoint(path, by_skill, args.batch_size, args.max_new_tokens)
        rec = {"tokens": tokens, "accuracy": {k: round(v, 4) for k, v in acc.items()},
               "per_item": per_item}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        pretty = "  ".join(f"{k}={v:.2f}" for k, v in sorted(acc.items()))
        print(f"tokens={tokens:>12,}  {pretty}")


if __name__ == "__main__":
    main()
