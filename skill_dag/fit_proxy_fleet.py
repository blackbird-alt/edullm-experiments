"""
Train the shared proxy fleet for arms 2 and 3. RUN ON GPU.

The doc: arm 2 uses "incomplete training runs of various 50-100M sized models and
extrapolation with the power laws to fit a LightGBM regression for val loss as a function
of domain weights"; arm 3 is "same as (2), but with the data mixing law". Same runs,
different regressor -- so ONE fleet serves both, and 2-vs-3 becomes a clean comparison of
regressor family rather than two unrelated pipelines.

What "incomplete" buys: each run is stopped early and its loss-vs-tokens curve recorded at
several points. Fitting E + B/S^a to that curve extrapolates the converged loss without
paying for convergence. Fitting the same form over model size extrapolates from 50-100M to
the 1B target. Both fitters read the identical curves from fleet.jsonl.

Mixtures are drawn from a Dirichlet centred on the NATURAL weights (alpha scaled by the
measured mix), which keeps them realistic rather than uniform over the simplex -- RegMix's
point: an unrealistic mixture teaches the regressor nothing useful about the region you
actually care about. The chosen weights must later be minimised WITHIN this support, or the
regressor is extrapolating.

Resumable: one line per completed run in fleet.jsonl; re-runs skip what is done.

Shardable: --shard I --num-shards N runs only every Nth run, so the 96 runs can be spread
over N GPUs instead of taken sequentially. Each worker appends to its own
fleet.shardNN.jsonl and the last one to finish folds them into the canonical fleet.jsonl
that fit_regmix.py and fit_mixing_law.py read; --assemble-only does that merge on its own.

Usage:
  python fit_proxy_fleet.py --out fleet --mixtures 32 --sizes 50 75 100
  python fit_proxy_fleet.py --out fleet --shard 3 --num-shards 8   # one worker of eight
  python fit_proxy_fleet.py --out fleet --assemble-only            # after the array finishes
"""
import argparse, json, os, time

import numpy as np
import torch

from fit_aij import (add_shard_args, make_proxy, merge_logs, read_logs, select_shard,
                     shard_log_path, total_cost, write_json_atomic)
from train_mixture import BASE_MODEL, BASE_REVISION, DomainPools, val_loss

HERE = os.path.dirname(os.path.abspath(__file__))

# hidden/layers pairs giving roughly the requested total parameter counts with the
# OLMo vocab (50304 x hidden dominates at these scales)
SIZE_PRESETS = {50: (512, 8), 75: (576, 9), 100: (640, 10)}


def sample_mixtures(n, natural, domains, alpha_scale, seed):
    """Dirichlet centred on the natural mix. Returns list of {domain: weight}."""
    rng = np.random.default_rng(seed)
    base = np.array([max(natural.get(d, 1e-6), 1e-6) for d in domains], dtype=float)
    base = base / base.sum()
    out = [dict(zip(domains, base))]                      # always include natural itself
    for _ in range(n - 1):
        w = rng.dirichlet(base * alpha_scale)
        out.append(dict(zip(domains, w.tolist())))
    return out


def train_curve(weights, hidden, layers, pools, args, device):
    """Train one incomplete run; record val loss at several points along the way."""
    model = make_proxy(args.base, args.revision, hidden, layers, args.init_seed, device)
    nparam = sum(p.numel() for p in model.parameters())
    # NO gradient checkpointing: these proxies are ~50-100M and fit easily, so
    # checkpointing would cost ~25-30% throughput across 141 probe/fleet runs for nothing.
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))

    marks = [int(args.fleet_tokens * f) for f in args.curve_points]
    curve, trained, t0, mi = [], 0, time.time(), 0
    while trained < args.fleet_tokens:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            ids, _ = pools.batch(weights, args.batch_size, args.seq_len)
            ids = ids.to(device)
            (model(input_ids=ids, labels=ids).loss / args.grad_accum).backward()
            trained += int(ids.numel())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if mi < len(marks) and trained >= marks[mi]:
            vl = val_loss(model, pools, args.seq_len, device, max_seqs=args.val_seqs)
            curve.append({"tokens": trained, "val_loss": vl})
            mi += 1

    if not curve or curve[-1]["tokens"] < trained:
        vl = val_loss(model, pools, args.seq_len, device, max_seqs=args.val_seqs)
        curve.append({"tokens": trained, "val_loss": vl})
    secs = time.time() - t0
    del model, opt
    torch.cuda.empty_cache()
    return curve, {"tokens": trained, "seconds": round(secs, 1), "params": nparam}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "dolma_domains"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--mixtures", type=int, default=32)
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 75, 100],
                    help="target proxy sizes in millions (doc: various 50-100M)")
    ap.add_argument("--alpha-scale", type=float, default=20.0,
                    help="Dirichlet concentration; higher = closer to natural")
    ap.add_argument("--fleet-tokens", type=int, default=200_000_000)
    ap.add_argument("--curve-points", type=float, nargs="+",
                    default=[0.2, 0.4, 0.6, 0.8, 1.0],
                    help="fractions of fleet-tokens at which to record val loss")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--revision", default=BASE_REVISION)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-seqs", type=int, default=32)
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=1234)
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="enable gradient checkpointing; off by default because the proxies "
                         "are small enough not to need it")
    ap.add_argument("--mixture-seed", type=int, default=7)
    add_shard_args(ap)
    args = ap.parse_args()

    for s in args.sizes:
        if s not in SIZE_PRESETS:
            raise SystemExit(f"no preset for {s}M; known: {sorted(SIZE_PRESETS)}")

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    pools = DomainPools(args.data, seed=args.data_seed)
    mixes = sample_mixtures(args.mixtures, pools.natural, pools.domains,
                            args.alpha_scale, args.mixture_seed)
    # every shard derives the same mixtures from the same seed, so this is written
    # atomically rather than guarded -- concurrent plain writes would tear the file
    write_json_atomic({"mixtures": mixes, "domains": pools.domains,
                       "alpha_scale": args.alpha_scale,
                       "mixture_seed": args.mixture_seed,
                       "note": "index 0 is the natural mix; all others Dirichlet around it"},
                      os.path.join(args.out, "mixtures.json"))

    tasks = [(mi, s) for mi in range(len(mixes)) for s in args.sizes]
    log_key = lambda r: (r["mixture_index"], r["size_m"])
    done = read_logs(args.out, "fleet", log_key)

    print(f"fleet: {len(mixes)} mixtures x {len(args.sizes)} sizes = {len(tasks)} runs "
          f"@ {args.fleet_tokens:,} tokens each")
    if args.num_shards > 1:
        print(f"shard {args.shard}/{args.num_shards}")
    if done:
        print(f"resuming: {len(done)}/{len(tasks)} runs already done")
    print()

    if not args.assemble_only:
        mine = select_shard(tasks, args.shard, args.num_shards)
        log_p = shard_log_path(args.out, "fleet", args.shard, args.num_shards)
        for n, (mi, s) in enumerate(mine, 1):
            if (mi, s) in done:
                continue
            hidden, layers = SIZE_PRESETS[s]
            top = sorted(mixes[mi].items(), key=lambda kv: -kv[1])[:3]
            print(f"[{n}/{len(mine)}] mix{mi} @{s}M  top: "
                  + ", ".join(f"{d}={w:.2f}" for d, w in top), flush=True)
            curve, cost = train_curve(mixes[mi], hidden, layers, pools, args, dev)
            rec = {"mixture_index": mi, "size_m": s, "hidden": hidden, "layers": layers,
                   "weights": mixes[mi], "curve": curve, "cost": cost}
            with open(log_p, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done[(mi, s)] = rec
            print(f"    final mean val loss {curve[-1]['val_loss']['_mean']:.4f} "
                  f"({cost['seconds']:.0f}s, {cost['params']/1e6:.0f}M params)", flush=True)
        done = read_logs(args.out, "fleet", log_key)   # pick up sibling shards

    # the regressors need the whole grid, so a shard that finishes early leaves the
    # canonical fleet.jsonl alone rather than publishing a partial one
    missing = [t for t in tasks if t not in done]
    if missing:
        print(f"\n{len(done)}/{len(tasks)} runs done, {len(missing)} outstanding "
              f"(next: mix{missing[0][0]} @{missing[0][1]}M).")
        print(f"Run --assemble-only once the rest finish to write {args.out}/fleet.jsonl.")
        return
    sources = merge_logs(args.out, "fleet", done)

    # summed over every run, including ones done in earlier or sibling jobs, so the
    # preregistered fitting-cost comparison is unaffected by how the work was split
    fit_seconds, fit_tokens = total_cost(done.values())
    write_json_atomic({"fitting_compute": {"seconds": fit_seconds, "tokens": fit_tokens,
                                           "gpu_hours": round(fit_seconds / 3600, 2)},
                       "n_runs": len(tasks), "config": vars(args),
                       "shard_logs": sources},
                      os.path.join(args.out, "fleet_cost.json"))
    print(f"\nfleet complete: {len(tasks)} runs, {fit_seconds/3600:.2f} GPU-h")
    print(f"-> {args.out}/fleet.jsonl  (read by fit_regmix.py and fit_mixing_law.py)")


if __name__ == "__main__":
    main()
