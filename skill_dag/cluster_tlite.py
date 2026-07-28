"""
T-LITE-style behavioral clustering of domains (arm 5). CPU only, seconds to run.

The doc's rule: represent each domain by a cheap BEHAVIORAL signature -- its estimated
effect on each skill (a row of A), or the data-mixing-law coefficients -- and explicitly
NOT by topic-embedding similarity. Then k-means into K clusters, and run Skill-It at
cluster level.

Why the signature source matters: arm 5 only earns its place if its probing is CHEAPER
than arm 4's. Deriving signatures from arm 4's full 45-probe A_ij would defeat the point.
So the default source is the arm-3 data-mixing-law fit, which is already computed for a
different arm and costs nothing extra here. `--from-aij` exists for diagnostics only --
comparing "clusters you would have found for free" against "clusters from the full
matrix" tells you whether the cheap signature was good enough.

Usage:
  python cluster_tlite.py --mixing-law mixlaw/fit.json --k 4 --out clusters.json
  python cluster_tlite.py --from-aij aij_arm4/aij.json --k 4 --out clusters_ref.json
"""
import argparse, json, math, os, random


def load_signatures(args):
    """-> ({domain: [floats]}, source_label)"""
    if args.from_aij:
        d = json.load(open(args.from_aij))
        A = d["A"]
        units = d["units"]
        # row of A = this domain's measured effect on every other domain
        return {i: [A[i][j] for j in units] for i in units}, "aij_rows"

    d = json.load(open(args.mixing_law))
    # Data-mixing law: loss_i = E_i + exp(sum_j t_ij * r_j). The per-domain coefficient
    # vector t_i is exactly "how this domain's weight moves each validation loss", i.e.
    # a behavioral signature obtained without any extra probing.
    coef = d.get("t") or d.get("coefficients") or d.get("t_ij")
    if not coef:
        raise SystemExit(f"{args.mixing_law} has no 't'/'coefficients' block")
    doms = sorted(coef)
    return {i: [float(coef[i][j]) for j in sorted(coef[i])] for i in doms}, "mixing_law_t"


def zscore(sig):
    """Standardise each signature dimension so no single column dominates the distance."""
    doms = sorted(sig)
    dim = len(sig[doms[0]])
    out = {d: list(v) for d, v in sig.items()}
    for c in range(dim):
        col = [sig[d][c] for d in doms]
        mu = sum(col) / len(col)
        sd = (sum((x - mu) ** 2 for x in col) / max(len(col) - 1, 1)) ** 0.5 or 1.0
        for d in doms:
            out[d][c] = (sig[d][c] - mu) / sd
    return out


def kmeans(sig, k, iters=200, restarts=25, seed=0):
    """Plain k-means with k-means++ seeding, best of `restarts` by inertia."""
    doms = sorted(sig)
    X = [sig[d] for d in doms]
    dim = len(X[0])
    if k >= len(doms):
        return {d: str(i) for i, d in enumerate(doms)}, 0.0

    def d2(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))

    best, best_inertia = None, math.inf
    for r in range(restarts):
        rng = random.Random(seed + r)
        cents = [list(X[rng.randrange(len(X))])]
        while len(cents) < k:                                  # k-means++
            dist = [min(d2(x, c) for c in cents) for x in X]
            tot = sum(dist) or 1.0
            pick, acc = 0, rng.random() * tot
            for idx, w in enumerate(dist):
                acc -= w
                if acc <= 0:
                    pick = idx
                    break
            cents.append(list(X[pick]))
        assign = [0] * len(X)
        for _ in range(iters):
            moved = False
            for i, x in enumerate(X):
                a = min(range(k), key=lambda c: d2(x, cents[c]))
                if a != assign[i]:
                    assign[i], moved = a, True
            for c in range(k):
                pts = [X[i] for i in range(len(X)) if assign[i] == c]
                if pts:
                    cents[c] = [sum(p[j] for p in pts) / len(pts) for j in range(dim)]
            if not moved:
                break
        inertia = sum(d2(X[i], cents[assign[i]]) for i in range(len(X)))
        if inertia < best_inertia:
            best_inertia, best = inertia, dict(zip(doms, assign))
    return {d: str(c) for d, c in best.items()}, best_inertia


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mixing-law", help="arm-3 fit json (the cheap signature source)")
    g.add_argument("--from-aij", help="arm-4 aij.json -- DIAGNOSTIC ONLY, not cheaper")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="clusters.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sig, source = load_signatures(args)
    clusters, inertia = kmeans(zscore(sig), args.k, seed=args.seed)

    groups = {}
    for d, c in clusters.items():
        groups.setdefault(c, []).append(d)
    k_actual = len(groups)
    probes_full = len(sig) + len(sig) * (len(sig) - 1) // 2
    probes_clust = k_actual + k_actual * (k_actual - 1) // 2

    json.dump({
        "clusters": clusters, "groups": groups, "k": args.k, "k_actual": k_actual,
        "signature_source": source, "inertia": round(inertia, 4),
        "probe_runs_full": probes_full, "probe_runs_clustered": probes_clust,
        "probe_reduction": round(1 - probes_clust / probes_full, 3),
        "note": ("Signatures are behavioral (effect on each validation loss), not topical, "
                 "per the T-LITE analogy. Within a cluster, training samples by natural "
                 "weighting."),
    }, open(args.out, "w"), indent=2)

    print(f"signature source: {source}   k={args.k} -> {k_actual} non-empty clusters")
    for c in sorted(groups):
        print(f"  cluster {c}: {', '.join(sorted(groups[c]))}")
    print(f"\nprobe runs: {probes_full} (full) -> {probes_clust} (clustered) "
          f"= {100*(1-probes_clust/probes_full):.0f}% fewer")
    if k_actual < args.k:
        print(f"WARNING: {args.k - k_actual} cluster(s) came out empty; k may be too high "
              f"for {len(sig)} domains")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
