"""
Continual pretraining with per-domain sampling weights. RUN ON GPU.

One invocation = one arm of the domain-mixture experiment. The script is identical
across arms; the domain weights (and whether they are updated mid-run) are the only
difference.

Arms (see PREREG.md):
  1 natural   --weights natural                        --reweight-mode fixed
  2 regmix    --weights weights_arm2.json              --reweight-mode fixed
  3 mixlaw    --weights weights_arm3.json              --reweight-mode fixed
  4 skill-it  --weights weights_arm4_init.json --aij aij.json --reweight-mode adaptive
  5 clustered --weights ... --aij aij_clustered.json --cluster-map clusters.json
              --reweight-mode adaptive

Design choices that protect the experiment (do not change casually):
- DEPENDENT VARIABLE IS VALIDATION LOSS. The hypothesis is about compute to reach a
  given val loss, so val loss on a held-out slice of every domain is logged on a
  fixed token cadence, and that curve is the primary output.
- 5 REWEIGHTING ROUNDS in adaptive mode (the spec's number), evenly spaced across
  the token budget. Round 0 uses the initial weights.
- HELD-OUT SLICES ARE NEVER TRAINED ON. The tail --val-frac of each domain pool is
  reserved before any sampling; the train sampler cannot reach it.
- CONSTANT LR + WARMUP by default. A decaying LR would down-weight whatever data
  arrives late, which in adaptive mode is exactly the data the reweighting chose --
  that would confound the fixed-vs-adaptive comparison. Override with --lr-schedule
  if you want to match a stock pretraining recipe instead.
- UNIQUE-TOKEN ACCOUNTING: each domain's read cursor advances monotonically and
  wraparound is counted and logged. wiki (2 shards) and books (3 shards) are small,
  so an arm that upweights them hard can exhaust its unique pool and start
  repeating -- that must show up in the logs, not silently.
- TRAINING COMPUTE IS LOGGED SEPARATELY from the fitting compute spent choosing the
  weights (see the fitter scripts' own cost records). The spec's success bar
  compares fitting cost across methods, so the two must not be pooled.

Usage:
  python train_mixture.py --arm arm1_natural --weights natural \
      --reweight-mode fixed --token-budget 5000000000 --out runs/arm1
"""
import argparse, json, math, os, signal, time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_constant_schedule_with_warmup

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "allenai/OLMo-1B-hf"
# Very early checkpoint: step 1000 of 738020 (~0.14% through pretraining).
# NOTE: sub-step-20000 revisions exist ONLY on the -hf repo, not allenai/OLMo-1B.
BASE_REVISION = "step1000-tokens4B"


class DomainPools:
    """Memmapped per-domain uint16 token pools with train/val split and cursors."""

    def __init__(self, data_dir, val_frac=0.005, seed=0, seed_index=0, n_seeds=1):
        """seed_index/n_seeds give each replicate a DISJOINT starting region.

        Without this every seed starts at position 0 and reads the same tokens,
        differing only in domain interleaving -- which makes the seeds not
        independent replicates and undercuts the run-level variance the
        preregistered statistical test needs.
        """
        mf = json.load(open(os.path.join(data_dir, "domain_manifest.json")))
        self.natural = mf["natural_weights"]
        self.domains = sorted(mf["domains"])
        self.pools, self.val, self.cursor, self.wraps = {}, {}, {}, {}
        self.start = {}
        for d in self.domains:
            arr = np.load(os.path.join(data_dir, mf["domains"][d]["path"]), mmap_mode="r")
            n_val = max(1, int(len(arr) * val_frac))
            self.pools[d] = arr[: len(arr) - n_val]   # train slice
            self.val[d] = np.asarray(arr[len(arr) - n_val:], dtype=np.int64)
            span = len(self.pools[d]) // max(n_seeds, 1)
            self.start[d] = (seed_index % max(n_seeds, 1)) * span
            self.cursor[d] = self.start[d]
            self.wraps[d] = 0
        self.seed_index, self.n_seeds = seed_index, n_seeds
        self.rng = np.random.default_rng(seed)

    def state(self):
        """Everything needed to resume mid-run."""
        return {"cursor": dict(self.cursor), "wraps": dict(self.wraps),
                "start": dict(self.start), "rng": self.rng.bit_generator.state}

    def load_state(self, st):
        self.cursor.update(st["cursor"])
        self.wraps.update(st["wraps"])
        self.rng.bit_generator.state = st["rng"]

    def take(self, dom, n):
        """Sequential read of n tokens from a domain, wrapping (and counting) at EOF.

        Always returns exactly n tokens. A pool shorter than n would otherwise yield a
        short row and blow up np.stack (or silently shrink the batch).
        """
        p = self.pools[dom]
        if len(p) < n:
            raise SystemExit(
                f"domain '{dom}' pool has {len(p):,} tokens < seq_len {n:,}; "
                f"rebuild the pool with a larger --tokens-per-domain")
        c = self.cursor[dom]
        if c + n > len(p):
            self.cursor[dom], c = 0, 0
            self.wraps[dom] += 1
        self.cursor[dom] = c + n
        return np.asarray(p[c: c + n], dtype=np.int64)

    def batch(self, weights, batch_size, seq_len):
        """Draw a batch, choosing each sequence's domain ~ weights."""
        doms = list(weights)
        probs = np.array([weights[d] for d in doms], dtype=np.float64)
        probs = probs / probs.sum()
        picks = self.rng.choice(len(doms), size=batch_size, p=probs)
        rows, per_dom = [], {}
        for i in picks:
            d = doms[i]
            rows.append(self.take(d, seq_len))
            per_dom[d] = per_dom.get(d, 0) + seq_len
        return torch.from_numpy(np.stack(rows)), per_dom

    def val_batches(self, dom, seq_len, max_seqs=32):
        v = self.val[dom]
        n = min(max_seqs, len(v) // seq_len)
        for i in range(n):
            yield torch.from_numpy(v[i * seq_len: (i + 1) * seq_len][None, :])


def save_ckpt(path, model, opt, sched, pools, prog):
    """Atomic-ish checkpoint: model/opt/sched tensors + all bookkeeping.

    Cursors and weights MUST be included -- without them a restart replays the
    same tokens and resets the reweighting schedule, silently changing the
    experiment rather than resuming it.
    """
    tmp = path + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    model.save_pretrained(tmp)
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict()},
               os.path.join(tmp, "optim.pt"))
    with open(os.path.join(tmp, "progress.json"), "w") as f:
        json.dump({"pools": pools.state(), **prog}, f, indent=2)
    # Never leave a window with no checkpoint on disk: stage the new one, swap the old
    # aside, promote, then delete. A crash mid-swap leaves either path or path.old valid.
    import shutil
    old = path + ".old"
    if os.path.exists(old):
        shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(path):
        os.replace(path, old)
    os.replace(tmp, path)
    shutil.rmtree(old, ignore_errors=True)


def load_ckpt(path, model, opt, sched, pools):
    """-> progress dict, or None if no usable checkpoint."""
    p = os.path.join(path, "progress.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        st = json.load(f)
    sd = torch.load(os.path.join(path, "optim.pt"), map_location="cpu", weights_only=False)
    opt.load_state_dict(sd["opt"])
    sched.load_state_dict(sd["sched"])
    pools.load_state(st.pop("pools"))
    return st


def load_weights(spec, pools):
    """--weights natural | path/to/weights.json -> {domain: weight}"""
    if spec == "natural":
        w = {d: pools.natural[d] for d in pools.domains if d in pools.natural}
    else:
        w = json.load(open(spec))
        w = w.get("weights", w)
        w = {d: float(v) for d, v in w.items() if d in pools.domains}
    missing = [d for d in pools.domains if d not in w]
    if missing:
        raise SystemExit(f"weights missing domains: {missing}")
    s = sum(w.values())
    return {d: v / s for d, v in w.items()}


@torch.no_grad()
def val_loss(model, pools, seq_len, device, max_seqs=32):
    """Mean per-domain CE loss on the held-out slice. This is the DV."""
    model.eval()
    out = {}
    for d in pools.domains:
        tot, nb = 0.0, 0
        for ids in pools.val_batches(d, seq_len, max_seqs):
            ids = ids.to(device)
            tot += float(model(input_ids=ids, labels=ids).loss)
            nb += 1
        out[d] = tot / max(nb, 1)
    model.train()
    out["_mean"] = sum(v for k, v in out.items() if not k.startswith("_")) / len(pools.domains)
    return out


def skillit_update(weights, aij, losses, eta, cluster_map=None, natural=None):
    """Skill-It online mirror descent.

    w_i <- w_i * exp(eta * sum_j max(A_ij, 0) * loss_j)

    A_ij is how much training on domain i helps skill/domain j; multiplying by the
    CURRENT loss on j means domains feeding still-unlearned skills get upweighted.
    Negative A entries are clipped (the spec notes the formula ignores them).

    With cluster_map, the update runs at CLUSTER level (K weights, not k) and each
    cluster's mass is then split across its members by natural weighting -- the
    T-LITE arm.
    """
    if cluster_map:
        clusters = sorted(set(cluster_map.values()))
        cw = {c: sum(weights[d] for d in weights if cluster_map[d] == c) for c in clusters}
        cl = {c: (sum(losses[d] for d in weights if cluster_map[d] == c)
                  / max(1, sum(1 for d in weights if cluster_map[d] == c))) for c in clusters}
        new_c = {}
        for ci in clusters:
            g = sum(max(aij.get(ci, {}).get(cj, 0.0), 0.0) * cl[cj] for cj in clusters)
            new_c[ci] = cw[ci] * math.exp(eta * g)
        z = sum(new_c.values()) or 1.0
        new_c = {c: v / z for c, v in new_c.items()}
        out = {}
        for c in clusters:
            members = [d for d in weights if cluster_map[d] == c]
            base = {d: (natural or {}).get(d, 1.0 / len(members)) for d in members}
            bs = sum(base.values()) or 1.0
            for d in members:
                out[d] = new_c[c] * base[d] / bs
        return out

    new = {}
    for i in weights:
        g = sum(max(aij.get(i, {}).get(j, 0.0), 0.0) * losses[j] for j in weights)
        new[i] = weights[i] * math.exp(eta * g)
    z = sum(new.values()) or 1.0
    return {d: v / z for d, v in new.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="label, e.g. arm1_natural")
    ap.add_argument("--data", default=os.path.join(HERE, "dolma_domains"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", required=True, help="'natural' or path to weights json")
    ap.add_argument("--reweight-mode", choices=["fixed", "adaptive"], required=True)
    ap.add_argument("--aij", default=None, help="A_ij json, required for adaptive")
    ap.add_argument("--cluster-map", default=None, help="{domain: cluster} json, arm 5")
    ap.add_argument("--rounds", type=int, default=5, help="reweighting rounds (spec: 5)")
    ap.add_argument("--eta", type=float, default=0.5, help="mirror-descent step size")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--revision", default=BASE_REVISION)
    ap.add_argument("--token-budget", type=int, required=True)
    ap.add_argument("--eval-tokens", type=int, default=250_000_000, help="val-loss cadence")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--lr-schedule", choices=["constant"], default="constant")
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-index", type=int, default=0,
                    help="replicate index; offsets this run's read position so seeds "
                         "read disjoint data")
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="total replicates per arm; pool is divided into this many regions")
    ap.add_argument("--ckpt-every-tokens", type=int, default=100_000_000,
                    help="resume-checkpoint cadence in tokens (0 disables)")
    # A token cadence alone is not enough on a cluster with short job windows: 100M tokens
    # is well over an hour even on fast hardware, so a short job would checkpoint once or
    # not at all and a preemption could cost hours. Whichever limit comes first saves.
    ap.add_argument("--ckpt-every-seconds", type=int, default=1800,
                    help="resume-checkpoint cadence in seconds (0 disables)")
    ap.add_argument("--max-seconds", type=int, default=0,
                    help="stop cleanly after this long, checkpointing first. Set it below "
                         "the Slurm time limit so the run ends on a good checkpoint "
                         "instead of being killed mid-write. 0 = no limit.")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <out>/ckpt_resume if present")
    ap.add_argument("--device", default="cuda",
                    help="cuda, or cpu for a tiny offline dry run of the control flow")
    ap.add_argument("--wandb", default=None)
    args = ap.parse_args()

    if args.reweight_mode == "adaptive" and not args.aij:
        raise SystemExit("--reweight-mode adaptive requires --aij")

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = args.device

    pools = DomainPools(args.data, val_frac=args.val_frac, seed=args.seed,
                        seed_index=args.seed_index, n_seeds=args.n_seeds)
    weights = load_weights(args.weights, pools)
    aij = json.load(open(args.aij)) if args.aij else {}
    aij = aij.get("A", aij)
    cmap = json.load(open(args.cluster_map)) if args.cluster_map else None
    if cmap:
        cmap = cmap.get("clusters", cmap)

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    # If resuming, model weights come from the checkpoint, NOT the base revision.
    ckpt_dir_pre = os.path.join(args.out, "ckpt_resume")
    will_resume = args.resume and os.path.exists(os.path.join(ckpt_dir_pre, "progress.json"))
    if will_resume:
        print(f"loading model from checkpoint {ckpt_dir_pre}")
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir_pre, torch_dtype=torch.bfloat16).to(dev)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, torch_dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False      # incompatible with checkpointing; silence + correctness
    model.train()

    wb = None
    if args.wandb:
        import wandb as wb
        parts = args.wandb.split("/")
        wb.init(entity=parts[0] if len(parts) > 1 else None, project=parts[-1],
                group="domain-mixture", name=args.arm, config=vars(args))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=args.warmup_steps)

    log_p = os.path.join(args.out, "train_log.jsonl")
    val_p = os.path.join(args.out, "val_log.jsonl")
    round_p = os.path.join(args.out, "weight_log.jsonl")
    tokens_per_round = args.token_budget // max(args.rounds, 1)

    def log_weights(rnd, tokens, w, losses=None):
        rec = {"round": rnd, "tokens": tokens, "weights": w, "val_loss": losses}
        with open(round_p, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[round {rnd}] tokens={tokens:,} weights=" +
              " ".join(f"{d}:{w[d]:.3f}" for d in sorted(w)), flush=True)

    # ---- resume, or start fresh ----
    ckpt_dir = os.path.join(args.out, "ckpt_resume")
    trained, step, next_eval, next_round = 0, 0, args.eval_tokens, tokens_per_round
    consumed = {d: 0 for d in pools.domains}
    resumed = None
    if will_resume:
        resumed = load_ckpt(ckpt_dir, model, opt, sched, pools)
    if resumed:
        # model weights themselves were reloaded from ckpt_dir below
        trained = resumed["trained"]; step = resumed["step"]
        next_eval = resumed["next_eval"]; next_round = resumed["next_round"]
        consumed = resumed["consumed"]; weights = resumed["weights"]
        print(f"RESUMED at {trained:,} tokens (step {step}); "
              f"weights=" + " ".join(f"{d}:{weights[d]:.3f}" for d in sorted(weights)))
    else:
        log_weights(0, 0, weights)
    print(f"arm={args.arm} mode={args.reweight_mode} base={args.model}@{args.revision}")
    print(f"budget={args.token_budget:,} rounds={args.rounds} "
          f"tokens/round={tokens_per_round:,}")
    print(f"seed={args.seed} seed_index={args.seed_index}/{args.n_seeds} "
          f"read offsets=" + " ".join(f"{d}:{pools.start[d]:,}" for d in pools.domains[:3])
          + " ...\n")

    next_ckpt = (trained + args.ckpt_every_tokens) if args.ckpt_every_tokens else float("inf")
    t0 = time.time()
    next_ckpt_t = (t0 + args.ckpt_every_seconds) if args.ckpt_every_seconds else float("inf")
    deadline = (t0 + args.max_seconds) if args.max_seconds else float("inf")

    # Slurm sends SIGTERM at the time limit and, with --signal, SIGUSR1 ahead of a
    # preemption. Catching them lets the step finish and a checkpoint land, instead of
    # losing everything since the last save. The flag is only read at a step boundary --
    # checkpointing from inside the handler could write a half-updated optimizer state.
    stop = {"why": None}

    def _stop(signum, _frame):
        stop["why"] = signal.Signals(signum).name

    for sig in (signal.SIGTERM, getattr(signal, "SIGUSR1", None)):
        if sig is not None:
            try:
                signal.signal(sig, _stop)
            except (ValueError, OSError):
                pass                        # not on the main thread, or unsupported

    def progress():
        return {"trained": trained, "step": step, "next_eval": next_eval,
                "next_round": next_round, "consumed": consumed, "weights": weights,
                "arm": args.arm}

    while trained < args.token_budget:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            ids, per_dom = pools.batch(weights, args.batch_size, args.seq_len)
            ids = ids.to(dev)
            loss = model(input_ids=ids, labels=ids).loss / args.grad_accum
            loss.backward()
            trained += int(ids.numel())
            for d, n in per_dom.items():
                consumed[d] += n
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1

        if step % 20 == 0:
            rec = {"step": step, "tokens": trained,
                   "loss": round(float(loss) * args.grad_accum, 4),
                   "elapsed_s": round(time.time() - t0)}
            with open(log_p, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if wb:
                wb.log(rec, step=step)

        if trained >= next_eval:
            vl = val_loss(model, pools, args.seq_len, dev)
            rec = {"tokens": trained, "step": step, "val_loss": vl,
                   "weights": dict(weights), "unique_wraps": dict(pools.wraps),
                   "domain_tokens_consumed": dict(consumed)}
            with open(val_p, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  val@{trained:,}: mean={vl['_mean']:.4f}", flush=True)
            if wb:
                wb.log({"val/" + k: v for k, v in vl.items()}, step=step)
            next_eval += args.eval_tokens

        if trained >= next_ckpt or time.time() >= next_ckpt_t:
            save_ckpt(ckpt_dir, model, opt, sched, pools, progress())
            print(f"  ckpt@{trained:,} -> {ckpt_dir}", flush=True)
            while trained >= next_ckpt:
                next_ckpt += args.ckpt_every_tokens
            if args.ckpt_every_seconds:
                next_ckpt_t = time.time() + args.ckpt_every_seconds

        if stop["why"] or time.time() >= deadline:
            why = stop["why"] or f"--max-seconds {args.max_seconds}"
            save_ckpt(ckpt_dir, model, opt, sched, pools, progress())
            print(f"\nSTOPPING EARLY ({why}) at {trained:,}/{args.token_budget:,} tokens "
                  f"({100 * trained / args.token_budget:.1f}%).")
            print(f"Checkpoint written to {ckpt_dir}. Resubmit the same command with "
                  f"--resume to continue; run_config.json is written only on completion, "
                  f"so its absence is how a launcher knows this run is unfinished.")
            return

        if args.reweight_mode == "adaptive" and trained >= next_round:
            rnd = int(trained // tokens_per_round)
            vl = val_loss(model, pools, args.seq_len, dev)
            losses = {d: vl[d] for d in pools.domains}
            weights = skillit_update(weights, aij, losses, args.eta,
                                     cluster_map=cmap, natural=pools.natural)
            log_weights(rnd, trained, weights, losses)
            next_round += tokens_per_round

    vl = json.dumps(val_loss(model, pools, args.seq_len, dev))
    with open(val_p, "a") as f:
        f.write(json.dumps({"tokens": trained, "step": step, "final": True,
                            "val_loss": json.loads(vl)}) + "\n")
    model.save_pretrained(os.path.join(args.out, "final"))
    tok.save_pretrained(os.path.join(args.out, "final"))
    json.dump(vars(args) | {
        "final_tokens": trained, "steps": step,
        "final_weights": weights, "domain_tokens_consumed": consumed,
        "unique_pool_wraps": pools.wraps,
        "read_start_offsets": pools.start,
        "tokens_per_step": args.batch_size * args.grad_accum * args.seq_len,
        "train_gpu_seconds": round(time.time() - t0),
    }, open(os.path.join(args.out, "run_config.json"), "w"), indent=2)
    print(f"\ndone: {trained:,} tokens, {step} steps -> {args.out}")
    if any(v for v in pools.wraps.values()):
        print(f"WARNING: unique pool exhausted and repeated: "
              f"{ {d: v for d, v in pools.wraps.items() if v} }")


if __name__ == "__main__":
    main()
