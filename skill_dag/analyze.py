"""
Analysis for the Skill-DAG weighting experiment. CPU only.

The preregistered question is about COMPUTE TO REACH A GIVEN VALIDATION LOSS, so the
primary output is tokens-to-target-loss per arm, evaluated across a RANGE of near-
convergence loss targets rather than one arbitrary cutoff. A single cutoff would let the
conclusion depend on where the cutoff was placed.

Three things this deliberately does:

- SUSTAINED CROSSING, not first crossing. A target counts as reached only when the loss is
  at or below it for `--consecutive` successive evaluations. First-crossing detection lets a
  single lucky evaluation set the tokens-to-target, which biases whichever arm happens to be
  noisier.
- NON-INFERIORITY, not superiority, where the doc asks for it. "Not statistically
  significantly worse" is a claim of sameness and needs a stated margin; there is no such
  thing as an unmargined equivalence test. --margin must be set before running.
- FITTING COMPUTE REPORTED SEPARATELY. Arms 2-5 all spend compute choosing weights before
  training starts. The doc's success bar for arms 5 and the derivative follow-up compares
  that fitting cost across methods, so pooling it into training compute would erase the
  quantity being tested.

Bootstrap resamples seeds within an arm (and, when >1 seed, is the only source of
run-level variance -- with 2 seeds the interval is barely meaningful and that is reported).

Usage:
  python analyze.py --runs runs/arm1_* runs/arm2_* runs/arm3_* runs/arm4_* runs/arm5_* \
      --margin 0.01 --out analysis.json
"""
import argparse, glob, json, os, random, re


def load_run(d):
    """-> {"arm":.., "seed":.., "curve":[(tokens, mean_val_loss)], "config":..}"""
    vp = os.path.join(d, "val_log.jsonl")
    if not os.path.exists(vp):
        return None
    pts = []
    for line in open(vp):
        r = json.loads(line)
        vl = r.get("val_loss") or {}
        if "_mean" in vl:
            pts.append((r["tokens"], float(vl["_mean"])))
    if not pts:
        return None
    pts.sort()
    cfg = {}
    cp = os.path.join(d, "run_config.json")
    if os.path.exists(cp):
        cfg = json.load(open(cp))
    name = os.path.basename(d.rstrip("/\\"))
    m = re.match(r"(arm\d+|[A-Za-z0-9]+?)[_-]?(?:seed)?(\d+)?$", name)
    arm = cfg.get("arm") or (m.group(1) if m else name)
    seed = cfg.get("seed_index", cfg.get("seed", m.group(2) if m and m.group(2) else 0))
    return {"dir": d, "arm": str(arm), "seed": int(seed or 0), "curve": pts, "config": cfg}


def tokens_to_target(curve, target, consecutive=2):
    """Tokens at which loss first drops to `target` and STAYS there for `consecutive`
    evaluations. Linear interpolation to the crossing. None if never sustained."""
    run = 0
    for i, (tok, loss) in enumerate(curve):
        if loss <= target:
            run += 1
            if run >= consecutive:
                j = i - run + 1                      # first point of the sustained window
                if j == 0:
                    return float(curve[0][0])
                t0, l0 = curve[j - 1]
                t1, l1 = curve[j]
                if l0 == l1:
                    return float(t1)
                frac = (l0 - target) / (l0 - l1)
                return t0 + frac * (t1 - t0)
        else:
            run = 0
    return None


def target_grid(runs, n, consecutive=2):
    """Loss targets spanning the region every run can actually SUSTAIN-cross.

    The upper bound is subtle. Using the worst run's FINAL loss looks natural but is wrong
    under sustained crossing: that run only touches its final loss at the last evaluation,
    so it can never hold it for `consecutive` points and is scored "not reached" at every
    target -- producing an empty comparison table.

    For a run to sustain-cross a target it needs `consecutive` evaluations at or below it,
    so the highest target it can satisfy is its loss at index len-consecutive. Taking the
    max of that across runs gives a ceiling every run can genuinely reach.
    """
    # Tightest target EVERY run can sustain-cross. For one run this is its loss at index
    # len-consecutive (the last value with `consecutive` points at or below it); across runs
    # the binding one is the maximum.
    tight = max(r["curve"][max(0, len(r["curve"]) - consecutive)][1] for r in runs)
    # Loosest useful target: what every run has already passed after its first few
    # evaluations. Anything above this is trivially reached and uninformative.
    loose = max(r["curve"][min(consecutive - 1, len(r["curve"]) - 1)][1] for r in runs)
    if loose <= tight:
        return [tight]
    return [tight + (loose - tight) * i / max(n - 1, 1) for i in range(n)]


def arm_mean(runs, arm, target, consecutive):
    vals = [tokens_to_target(r["curve"], target, consecutive)
            for r in runs if r["arm"] == arm]
    got = [v for v in vals if v is not None]
    if not got:
        return None, len(vals), 0
    return sum(got) / len(got), len(vals), len(got)


def bootstrap_ratio(runs, arm_a, arm_b, target, consecutive, iters, rng):
    """CI on mean_tokens(arm_b)/mean_tokens(arm_a) by resampling seeds within each arm."""
    A = [r for r in runs if r["arm"] == arm_a]
    B = [r for r in runs if r["arm"] == arm_b]
    if not A or not B:
        return None
    out = []
    for _ in range(iters):
        sa = [A[rng.randrange(len(A))] for _ in A]
        sb = [B[rng.randrange(len(B))] for _ in B]
        va = [tokens_to_target(r["curve"], target, consecutive) for r in sa]
        vb = [tokens_to_target(r["curve"], target, consecutive) for r in sb]
        va = [v for v in va if v is not None]
        vb = [v for v in vb if v is not None]
        if not va or not vb:
            continue
        ma, mb = sum(va) / len(va), sum(vb) / len(vb)
        if ma > 0:
            out.append(mb / ma)
    if len(out) < iters * 0.5:
        return None
    out.sort()
    return {"lo": out[int(0.025 * len(out))], "hi": out[int(0.975 * len(out))],
            "median": out[len(out) // 2], "n_valid": len(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories (globs ok)")
    ap.add_argument("--margin", type=float, required=True,
                    help="non-inferiority margin as a FRACTION of compute, e.g. 0.05 = "
                         "'within 5%% is not worse'. Must be preregistered.")
    ap.add_argument("--consecutive", type=int, default=2,
                    help="evaluations a target must be held for to count as reached")
    ap.add_argument("--targets", type=int, default=8, help="loss targets to sweep")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fitting-costs", default=None,
                    help="json {arm: gpu_hours} from the fitters, reported separately")
    ap.add_argument("--baseline-arm", default=None,
                    help="arm treated as the fixed-weight reference (default: lowest arm id)")
    ap.add_argument("--out", default="analysis.json")
    args = ap.parse_args()

    dirs = []
    for pat in args.runs:
        dirs.extend(sorted(glob.glob(pat)) or [pat])
    runs = [r for r in (load_run(d) for d in dirs) if r]
    if not runs:
        raise SystemExit("no runs with a usable val_log.jsonl")
    arms = sorted({r["arm"] for r in runs})
    print(f"loaded {len(runs)} runs across arms {arms}")
    for a in arms:
        seeds = sorted(r["seed"] for r in runs if r["arm"] == a)
        print(f"  {a}: {len(seeds)} seed(s) {seeds}")
    thin = [a for a in arms if sum(1 for r in runs if r["arm"] == a) < 3]
    if thin:
        print(f"  NOTE: {thin} have <3 seeds; run-level bootstrap intervals there are "
              f"weakly identified")

    grid = target_grid(runs, args.targets, args.consecutive)
    print(f"\nloss targets swept: {min(grid):.4f} .. {max(grid):.4f} ({len(grid)} points)")
    print(f"sustained crossing: {args.consecutive} consecutive evaluations")

    base = args.baseline_arm or arms[0]
    rng = random.Random(args.seed)
    rows, comparisons = [], []
    for t in grid:
        row = {"target": t, "arms": {}}
        for a in arms:
            m, n, reached = arm_mean(runs, a, t, args.consecutive)
            row["arms"][a] = {"mean_tokens": m, "seeds": n, "seeds_reaching": reached}
        rows.append(row)
        for a in arms:
            if a == base:
                continue
            ci = bootstrap_ratio(runs, base, a, t, args.consecutive, args.boot, rng)
            mb = row["arms"][base]["mean_tokens"]
            ma = row["arms"][a]["mean_tokens"]
            if mb and ma and ci:
                comparisons.append({
                    "target": t, "arm": a, "vs": base,
                    "ratio": ma / mb, "ci": ci,
                    # non-inferior if the CI upper bound stays inside the margin
                    "non_inferior": ci["hi"] <= 1.0 + args.margin,
                    "superior": ci["hi"] < 1.0,
                })

    print(f"\n=== mean tokens to reach each loss target ===")
    print(f"{'target':>9} " + " ".join(f"{a:>13}" for a in arms))
    for row in rows:
        cells = []
        for a in arms:
            m = row["arms"][a]["mean_tokens"]
            r = row["arms"][a]
            cells.append(f"{m/1e6:>10.0f}M" if m else
                         f"{'not reached':>13}" if r["seeds_reaching"] == 0 else
                         f"{'partial':>13}")
        print(f"{row['target']:>9.4f} " + " ".join(f"{c:>13}" for c in cells))

    print(f"\n=== vs {base} (ratio <1 means fewer tokens; margin {args.margin:.0%}) ===")
    for a in arms:
        cs = [c for c in comparisons if c["arm"] == a]
        if not cs:
            continue
        ni = sum(1 for c in cs if c["non_inferior"])
        sup = sum(1 for c in cs if c["superior"])
        med = sorted(c["ratio"] for c in cs)[len(cs) // 2]
        print(f"  {a:>8}: median ratio {med:5.3f}  non-inferior at {ni}/{len(cs)} targets, "
              f"superior at {sup}/{len(cs)}")

    fitting = json.load(open(args.fitting_costs)) if args.fitting_costs else {}
    if fitting:
        print("\n=== fitting compute (NOT pooled with training) ===")
        for a in sorted(fitting):
            print(f"  {a:>8}: {fitting[a]} GPU-h to choose its weights")

    json.dump({
        "arms": arms, "baseline": base, "n_runs": len(runs),
        "seeds_per_arm": {a: sum(1 for r in runs if r["arm"] == a) for a in arms},
        "margin": args.margin, "consecutive": args.consecutive,
        "targets": grid, "per_target": rows, "comparisons": comparisons,
        "fitting_compute_gpu_hours": fitting,
        "notes": [
            "tokens-to-target uses SUSTAINED crossing; a single lucky evaluation cannot "
            "set the value",
            "non-inferiority is judged by the bootstrap CI upper bound against the "
            "preregistered margin, not by a point estimate",
            "fitting compute is reported separately because the doc's success bar for "
            "arm 5 and the derivative follow-up compares fitting cost across methods",
            "arms with <3 seeds have weakly identified run-level intervals",
        ],
    }, open(args.out, "w"), indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
