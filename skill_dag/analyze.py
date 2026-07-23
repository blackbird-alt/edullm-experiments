"""
Analysis + preregistered decision rule for the skill-DAG experiment. CPU.

Inputs: run directories (3 topological, 3 random), each containing the
eval_log.jsonl written by eval_mastery.py.

Computes, exactly as preregistered:
- tokens-to-mastery per skill per run (first token count where held-out accuracy
  crosses the mastery threshold; linear interpolation between checkpoints;
  censored if never reached within budget)
- E_tokens = mean(random tokens-to-mastery) / mean(topo tokens-to-mastery),
  aggregated across skills by geometric mean (correct aggregation for ratios)
- 95% bootstrap CI on E_tokens: each iteration resamples runs (with replacement,
  within arm) AND held-out items (with replacement, per skill), recomputing the
  accuracy curves and crossings from the stored per-item hits
- floor guard: topo's final accuracy may not trail random's by more than
  --floor-epsilon on any skill

Decision (prereg): KEEP if CI lower bound > 1.05 and floor guard passes.
KILL if CI upper bound <= 1.00 or floor guard fails. Else INCONCLUSIVE.

Usage:
  python analyze.py --topo runs/topo_101 runs/topo_102 runs/topo_103 \
                    --random runs/random_201 runs/random_202 runs/random_203
"""
import argparse, json, math, os, random

MASTERY = 0.90
BOOT_ITERS = 1000
SEED = 7


def load_run(run_dir):
    """-> sorted list of (tokens, {skill: [item hits]})"""
    path = os.path.join(run_dir, "eval_log.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["tokens"], r["per_item"]))
    rows.sort()
    assert rows, f"empty eval log: {path}"
    return rows


def curve_from_items(rows, skill, item_idx=None):
    """Accuracy-vs-tokens curve for one skill, optionally over a resampled item set."""
    pts = []
    for tokens, per_item in rows:
        hits = per_item[skill]
        if item_idx is not None:
            hits = [hits[i] for i in item_idx]
        pts.append((tokens, sum(hits) / len(hits)))
    return pts


def tokens_to_mastery(curve, threshold):
    """First token count where the curve crosses threshold (linear interp); None if never."""
    prev_t, prev_a = 0, 0.0
    for t, a in curve:
        if a >= threshold:
            if a == prev_a:
                return t
            frac = (threshold - prev_a) / (a - prev_a)
            return prev_t + frac * (t - prev_t)
        prev_t, prev_a = t, a
    return None  # censored


def e_tokens(topo_runs, rand_runs, skills, threshold, item_samples=None):
    """Geometric-mean E_tokens across skills; returns (value, censored_skills)."""
    ratios, censored = [], []
    for sk in skills:
        idx = item_samples.get(sk) if item_samples else None
        topo_vals = [tokens_to_mastery(curve_from_items(r, sk, idx), threshold) for r in topo_runs]
        rand_vals = [tokens_to_mastery(curve_from_items(r, sk, idx), threshold) for r in rand_runs]
        if any(v is None for v in topo_vals + rand_vals):
            censored.append(sk)
            continue
        ratios.append((sum(rand_vals) / len(rand_vals)) / (sum(topo_vals) / len(topo_vals)))
    if not ratios:
        return None, censored
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios)), censored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", nargs="+", required=True)
    ap.add_argument("--random", nargs="+", required=True)
    ap.add_argument("--mastery", type=float, default=MASTERY)
    ap.add_argument("--floor-epsilon", type=float, default=0.02)
    args = ap.parse_args()
    thr = args.mastery

    topo_runs = [load_run(d) for d in args.topo]
    rand_runs = [load_run(d) for d in args.random]
    skills = sorted(topo_runs[0][0][1].keys())
    n_items = {sk: len(topo_runs[0][0][1][sk]) for sk in skills}

    # ---- point estimate ----
    est, censored = e_tokens(topo_runs, rand_runs, skills, thr)
    print(f"skills: {skills}")
    if censored:
        print(f"CENSORED (mastery never reached in >=1 run; excluded from ratio): {censored}")
    print(f"E_tokens point estimate (geometric mean over skills): "
          f"{est:.4f}" if est else "E_tokens undefined: all skills censored")

    # per-skill table
    print(f"\n{'skill':6} {'topo mean tok-to-mastery':>26} {'random mean':>14} {'ratio':>7}")
    for sk in skills:
        tv = [tokens_to_mastery(curve_from_items(r, sk), thr) for r in topo_runs]
        rv = [tokens_to_mastery(curve_from_items(r, sk), thr) for r in rand_runs]
        if any(v is None for v in tv + rv):
            print(f"{sk:6} {'censored':>26}")
            continue
        tm, rm = sum(tv) / len(tv), sum(rv) / len(rv)
        print(f"{sk:6} {tm:>26,.0f} {rm:>14,.0f} {rm / tm:>7.3f}")

    # ---- bootstrap ----
    rng = random.Random(SEED)
    boots = []
    for _ in range(BOOT_ITERS):
        t_rs = [topo_runs[rng.randrange(len(topo_runs))] for _ in topo_runs]
        r_rs = [rand_runs[rng.randrange(len(rand_runs))] for _ in rand_runs]
        item_samples = {sk: [rng.randrange(n_items[sk]) for _ in range(n_items[sk])] for sk in skills}
        v, _ = e_tokens(t_rs, r_rs, skills, thr, item_samples)
        if v is not None:
            boots.append(v)
    boots.sort()
    if len(boots) < BOOT_ITERS * 0.5:
        print(f"\nWARNING: only {len(boots)}/{BOOT_ITERS} bootstrap iterations produced a "
              f"defined E_tokens (heavy censoring); CI is unreliable")
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    print(f"\n95% bootstrap CI on E_tokens: [{lo:.4f}, {hi:.4f}]  ({len(boots)} valid iters)")

    # ---- floor guard ----
    floor_fail = []
    for sk in skills:
        topo_final = sum(curve_from_items(r, sk)[-1][1] for r in topo_runs) / len(topo_runs)
        rand_final = sum(curve_from_items(r, sk)[-1][1] for r in rand_runs) / len(rand_runs)
        if topo_final < rand_final - args.floor_epsilon:
            floor_fail.append((sk, round(topo_final, 3), round(rand_final, 3)))
    print(f"floor guard (topo final >= random final - {args.floor_epsilon}): "
          f"{'PASS' if not floor_fail else f'FAIL {floor_fail}'}")

    # ---- preregistered decision ----
    if floor_fail or hi <= 1.00:
        verdict = "KILL"
    elif lo > 1.05:
        verdict = "KEEP"
    else:
        verdict = "INCONCLUSIVE"
    print(f"\nVERDICT (prereg rule): {verdict}")
    print("  KEEP requires CI lower bound > 1.05 and floor guard PASS")
    print("  KILL on harm, floor failure, or CI upper bound <= 1.00")

    with open("analysis_result.json", "w") as f:
        json.dump({"e_tokens": est, "ci": [lo, hi], "censored": censored,
                   "floor_fail": floor_fail, "verdict": verdict,
                   "mastery_threshold": thr, "boot_iters": len(boots)}, f, indent=2)


if __name__ == "__main__":
    main()
