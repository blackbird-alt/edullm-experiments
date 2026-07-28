"""
Estimate the Skill-It dependency matrix A_ij by pairwise probing. RUN ON GPU.

A_ij = how much training on domain i improves domain j. This IS the skill DAG that
arms 4 and 5 use to set weights.

Method (the doc's "isolating each pair"):
  loss_j(j alone)          <- 9 single-domain probes
  loss_j({i,j} co-trained) <- 36 pair probes
  A_ij = loss_j(j alone) - loss_j({i,j})        positive => i helps j

The training SET {i,j} is unordered, so ONE pair probe yields both A_ij and A_ji by
evaluating on both domains. That is 9 + C(9,2) = 45 runs, not 9 + 9*8 = 81.

Every probe starts from the SAME freshly-initialised proxy (fixed seed) so differences
are attributable to the data, not the init. Proxy defaults to ~100M params built from the
base model's own config -- same tokenizer and vocab as the main runs, which matters
because token IDs must mean the same thing.

Arm 5 (T-LITE) runs the identical procedure at CLUSTER level via --cluster-map: K
singles + C(K,2) pairs. For K=4 that is 10 runs vs 45 -- the compute lever that justifies
the arm. Fitting compute is recorded separately from training compute because the
preregistered success bar compares fitting cost across methods.

Resumable: each probe's result is appended to <out>/probes.jsonl and re-runs skip
completed probes.

Usage:
  python fit_aij.py --out aij_arm4                        # 45 runs, domain level
  python fit_aij.py --out aij_arm5 --cluster-map clusters.json   # 10 runs, cluster level
"""
import argparse, itertools, json, os, time

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from train_mixture import BASE_MODEL, BASE_REVISION, DomainPools, val_loss

HERE = os.path.dirname(os.path.abspath(__file__))


def make_proxy(base, revision, hidden, layers, seed, device):
    """Fresh small model from the base architecture. Same vocab/tokenizer as main runs."""
    torch.manual_seed(seed)
    cfg = AutoConfig.from_pretrained(base, revision=revision)
    cfg.hidden_size = hidden
    cfg.num_hidden_layers = layers
    cfg.intermediate_size = 4 * hidden
    cfg.num_attention_heads = max(1, hidden // 64)
    # only touch KV heads if the architecture actually has them; blindly assigning can
    # set the field to None on configs that omit it and break model construction
    if getattr(cfg, "num_key_value_heads", None):
        cfg.num_key_value_heads = cfg.num_attention_heads
    model = AutoModelForCausalLM.from_config(cfg)
    return model.to(device=device, dtype=torch.bfloat16)


def probe(train_units, unit_members, pools, args, device):
    """Train a fresh proxy on a uniform mix of `train_units`; return per-domain val loss.

    unit_members maps a unit (domain or cluster) -> list of underlying domains. Within a
    unit, sampling follows natural weighting (the doc's rule for cluster-level runs).
    """
    weights = {}
    for u in train_units:
        members = unit_members[u]
        nat = {d: pools.natural.get(d, 1.0 / len(members)) for d in members}
        s = sum(nat.values()) or 1.0
        for d in members:
            weights[d] = (1.0 / len(train_units)) * nat[d] / s

    model = make_proxy(args.base, args.revision, args.hidden, args.layers,
                       args.init_seed, device)
    nparam = sum(p.numel() for p in model.parameters())
    # NO gradient checkpointing: these proxies are ~50-100M and fit easily, so
    # checkpointing would cost ~25-30% throughput across 141 probe/fleet runs for nothing.
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))

    trained, t0 = 0, time.time()
    while trained < args.probe_tokens:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            ids, _ = pools.batch(weights, args.batch_size, args.seq_len)
            ids = ids.to(device)
            (model(input_ids=ids, labels=ids).loss / args.grad_accum).backward()
            trained += int(ids.numel())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    vl = val_loss(model, pools, args.seq_len, device, max_seqs=args.val_seqs)
    secs = time.time() - t0
    del model, opt
    torch.cuda.empty_cache()
    return vl, {"tokens": trained, "seconds": round(secs, 1), "params": nparam}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "dolma_domains"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cluster-map", default=None,
                    help="{domain: cluster} json -> probe at cluster level (arm 5)")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--revision", default=BASE_REVISION)
    ap.add_argument("--hidden", type=int, default=640, help="proxy hidden size (~100M total)")
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--probe-tokens", type=int, default=200_000_000)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-seqs", type=int, default=32)
    ap.add_argument("--init-seed", type=int, default=0,
                    help="fixed across probes so results are comparable")
    ap.add_argument("--data-seed", type=int, default=1234)
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="enable gradient checkpointing; off by default because the proxies "
                         "are small enough not to need it")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    pools = DomainPools(args.data, seed=args.data_seed)

    # units = domains, or clusters if a cluster map is given
    if args.cluster_map:
        cm = json.load(open(args.cluster_map))
        cm = cm.get("clusters", cm)
        unit_members = {}
        for d, c in cm.items():
            unit_members.setdefault(str(c), []).append(d)
        missing = [d for d in pools.domains if d not in cm]
        if missing:
            raise SystemExit(f"cluster map missing domains: {missing}")
        level = "cluster"
    else:
        unit_members = {d: [d] for d in pools.domains}
        level = "domain"
    units = sorted(unit_members)
    k = len(units)

    tasks = [("single", (u,)) for u in units] + \
            [("pair", p) for p in itertools.combinations(units, 2)]
    print(f"level={level}  units={k}  probes={len(tasks)} "
          f"({k} singles + {k*(k-1)//2} pairs)")
    print(f"probe_tokens={args.probe_tokens:,}  proxy hidden={args.hidden} "
          f"layers={args.layers}\n")

    # resume
    done, log_p = {}, os.path.join(args.out, "probes.jsonl")
    if os.path.exists(log_p):
        for line in open(log_p):
            r = json.loads(line)
            done[tuple(r["units"])] = r
        print(f"resuming: {len(done)}/{len(tasks)} probes already done\n")

    fit_seconds, fit_tokens = 0.0, 0
    for n, (kind, us) in enumerate(tasks, 1):
        if tuple(us) in done:
            fit_seconds += done[tuple(us)]["cost"]["seconds"]
            fit_tokens += done[tuple(us)]["cost"]["tokens"]
            continue
        print(f"[{n}/{len(tasks)}] {kind}: {'+'.join(us)}", flush=True)
        vl, cost = probe(list(us), unit_members, pools, args, dev)
        rec = {"kind": kind, "units": list(us), "val_loss": vl, "cost": cost}
        with open(log_p, "a") as f:
            f.write(json.dumps(rec) + "\n")
        done[tuple(us)] = rec
        fit_seconds += cost["seconds"]; fit_tokens += cost["tokens"]
        print(f"    mean val loss {vl['_mean']:.4f}  ({cost['seconds']:.0f}s)", flush=True)

    # ---- assemble A ----
    def unit_loss(rec, u):
        """Mean val loss over the domains making up unit u."""
        ms = unit_members[u]
        return sum(rec["val_loss"][d] for d in ms) / len(ms)

    alone = {u: unit_loss(done[(u,)], u) for u in units}
    A = {i: {j: 0.0 for j in units} for i in units}
    for i, j in itertools.combinations(units, 2):
        rec = done[(i, j)]
        # i helps j if co-training lowers j's loss vs j alone
        A[i][j] = alone[j] - unit_loss(rec, j)
        A[j][i] = alone[i] - unit_loss(rec, i)

    # Normalise by max |A| so the mirror-descent eta is interpretable across runs.
    mx = max((abs(v) for row in A.values() for v in row.values()), default=0.0) or 1.0
    A_norm = {i: {j: A[i][j] / mx for j in A[i]} for i in A}

    pos = sum(1 for r in A_norm.values() for v in r.values() if v > 0)
    out = {
        "level": level, "units": units, "unit_members": unit_members,
        "A": A_norm, "A_raw": A, "normaliser": mx, "loss_alone": alone,
        "n_probes": len(tasks),
        "fitting_compute": {"seconds": round(fit_seconds, 1), "tokens": fit_tokens,
                            "gpu_hours": round(fit_seconds / 3600, 2),
                            "note": "fitting only; excluded from main-run training compute"},
        "config": vars(args),
    }
    json.dump(out, open(os.path.join(args.out, "aij.json"), "w"), indent=2)

    print(f"\n=== A ({level} level, normalised) ===")
    w = max(len(u) for u in units)
    print(" " * (w + 2) + " ".join(f"{u[:7]:>8}" for u in units))
    for i in units:
        print(f"  {i:<{w}} " + " ".join(f"{A_norm[i][j]:>8.3f}" for j in units))
    print(f"\npositive entries: {pos}/{k*k}  (all-zero or all-equal A means the probes "
          f"were too short to measure transfer -- see review item 5)")
    print(f"fitting compute: {fit_seconds/3600:.2f} GPU-h over {len(tasks)} probes")
    print(f"-> {args.out}/aij.json")


if __name__ == "__main__":
    main()
