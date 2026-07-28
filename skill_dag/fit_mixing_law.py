"""
Arm 3: pick FIXED domain weights via the data-mixing law. CPU only.

The doc: "Same as (2), but with the data mixing law", with the functional form
    L_i(r) = c_i + k_i * exp( sum_j t_ij * r_j )
where t_ij, k_i, c_i are fitted and r_j are the domain weights.

Same fleet, same power-law extrapolation as arm 2 -- only the regressor differs. That is
what makes 2-vs-3 a comparison of regressor family (nonparametric boosted trees vs this
parametric form) rather than two unrelated pipelines.

Fitting without SciPy: for a fixed c_i the model linearises,
    log(L_i - c_i) = log k_i + sum_j t_ij r_j
which is ordinary least squares in (log k_i, t_i.). So c_i is scanned on a grid below the
observed minimum and the best least-squares solution kept. Reported r^2 is on the original
loss scale, not the log scale, so it is comparable to arm 2's RMSE.

Also writes the fitted `t` matrix, which is the cheap behavioural signature
`cluster_tlite.py` consumes for arm 5 -- available at no extra probing cost, which is
precisely why arm 5 can be cheaper than arm 4.

Usage:
  python fit_mixing_law.py --fleet fleet --out weights_arm3.json --t-out mixlaw_t.json
"""
import argparse, json, os

import numpy as np

from extrapolate import converged_loss, extrapolate_to_size, select_in_support
from fit_regmix import TARGET_PARAMS, load_fleet


def fit_one(R, L, c_grid=80):
    """Fit L = c + k*exp(R @ t) for one validation target.

    Returns (c, log_k, t, r2) with r2 measured on the loss scale.
    """
    lo = float(L.min())
    best = None
    for i in range(c_grid):
        # c must sit strictly below the smallest observed loss for the log to exist
        c = lo * (0.05 + 0.90 * i / (c_grid - 1))
        z = L - c
        if np.any(z <= 0):
            continue
        A = np.hstack([R, np.ones((len(R), 1))])
        sol, *_ = np.linalg.lstsq(A, np.log(z), rcond=None)
        t, logk = sol[:-1], sol[-1]
        pred = c + np.exp(A @ np.concatenate([t, [logk]]))
        sse = float(((L - pred) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, c, logk, t)
    if best is None:
        return float(L.mean()), 0.0, np.zeros(R.shape[1]), 0.0
    sse, c, logk, t = best
    tss = float(((L - L.mean()) ** 2).sum()) or 1e-18
    return c, float(logk), t, 1.0 - sse / tss


def predict_one(R, c, logk, t):
    return c + np.exp(R @ t + logk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", required=True)
    ap.add_argument("--out", default="weights_arm3.json")
    ap.add_argument("--t-out", default="mixlaw_t.json",
                    help="fitted t matrix; the signature source for cluster_tlite.py")
    ap.add_argument("--target-params", type=float, default=TARGET_PARAMS)
    ap.add_argument("--candidates", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--allow-extrapolation", action="store_true",
                    help="permit predictions below every observed loss (off by default)")
    args = ap.parse_args()

    runs = load_fleet(args.fleet)
    domains = sorted(runs[0]["weights"])
    print(f"fleet: {len(runs)} runs, {len(domains)} domains")

    # Fit the law per validation domain (needed for the t matrix), plus the mean.
    targets = domains + ["_mean"]
    per_mix = {}
    diag = {"curve_fallback": 0, "size_fallback": 0}
    for key in targets:
        for r in runs:
            E, info = converged_loss(r["curve"], key)
            if E is None:
                continue
            if not info["ok"]:
                diag["curve_fallback"] += 1
            per_mix.setdefault(key, {}).setdefault(
                r["mixture_index"], {"weights": r["weights"], "sizes": {}}
            )["sizes"][float(r["cost"]["params"])] = E

    fits, R_ref, ys_ref = {}, None, None
    for key in targets:
        rows, ys = [], []
        for mi, rec in sorted(per_mix.get(key, {}).items()):
            pred, info = extrapolate_to_size(rec["sizes"], args.target_params)
            if pred is None:
                continue
            if not info["ok"]:
                diag["size_fallback"] += 1
            rows.append([rec["weights"][d] for d in domains])
            ys.append(pred)
        if len(rows) < len(domains) + 2:
            print(f"  {key:14} SKIPPED: {len(rows)} mixtures < {len(domains)+2} params")
            continue
        R, L = np.array(rows), np.array(ys)
        c, logk, t, r2 = fit_one(R, L)
        fits[key] = {"c": c, "log_k": logk, "t": t.tolist(), "r2": r2, "n": len(rows)}
        print(f"  {key:14} r2={r2:6.3f}  c={c:7.4f}  n={len(rows)}")
        if key == "_mean":
            R_ref, ys_ref = R, ys

    if "_mean" not in fits:
        raise SystemExit("could not fit the aggregate law; need more fleet mixtures")

    # minimise within the fleet's Dirichlet support
    mixmeta = json.load(open(os.path.join(args.fleet, "mixtures.json")))
    alpha_scale = mixmeta.get("alpha_scale", 20.0)
    natural = np.array([mixmeta["mixtures"][0][d] for d in domains], dtype=float)
    natural = natural / natural.sum()
    rng = np.random.default_rng(args.seed)
    C = rng.dirichlet(natural * alpha_scale, size=args.candidates)
    C = np.vstack([C, natural[None, :]])
    f = fits["_mean"]
    pred = predict_one(C, f["c"], f["log_k"], np.array(f["t"]))
    y_ref = predict_one(R_ref, f["c"], f["log_k"], np.array(f["t"]))
    best, sel = select_in_support(C, R_ref, np.array(ys), pred,
                                 allow_extrapolation=args.allow_extrapolation)
    if best is None:
        best = int(np.argmin(ys_ref))
        w = R_ref[best] / R_ref[best].sum()
        sel["selection"] = "fell back to the best OBSERVED fleet mixture"
        pred_best = float(ys_ref[best])
    else:
        w = C[best] / C[best].sum()
        pred_best = float(pred[best])
    print(f"  candidate search: {sel['in_box']}/{sel['candidates']} inside the fleet's "
          f"per-domain range, {sel['eligible']} eligible")
    if sel["raw_best_below_floor"]:
        print(f"  NOTE: unconstrained argmin predicted {sel['raw_best_prediction']:.4f}, "
              f"below the observed floor {sel['observed_loss_floor']:.4f} -- rejected as "
              f"law extrapolation")
    nat_pred = float(pred[-1])

    json.dump({
        "arm": 3, "method": "power-law + data-mixing law",
        "weights": {d: float(w[i]) for i, d in enumerate(domains)},
        "predicted_loss": pred_best,
        "predicted_loss_natural": nat_pred,
        "predicted_improvement_vs_natural": nat_pred - pred_best,
        "law": {"form": "L_i(r) = c_i + k_i * exp(sum_j t_ij r_j)",
                "aggregate_fit": {k: v for k, v in f.items() if k != "t"},
                "aggregate_t": dict(zip(domains, f["t"]))},
        "per_target_r2": {k: v["r2"] for k, v in fits.items()},
        "extrapolation_diagnostics": diag,
        "candidates_searched": len(C), "alpha_scale": alpha_scale,
        "support_selection": sel,
        "config": vars(args),
    }, open(args.out, "w"), indent=2)

    # t matrix for arm 5's clustering: row i = how each domain's weight moves loss i
    tmat = {i: dict(zip(domains, fits[i]["t"])) for i in domains if i in fits}
    json.dump({"t": tmat, "domains": domains,
               "r2": {i: fits[i]["r2"] for i in tmat},
               "note": ("row i = fitted t_ij, i.e. how domain j's weight moves validation "
                        "loss i. Behavioural signature for cluster_tlite.py; costs no extra "
                        "probing because arm 3 needed this fit anyway.")},
              open(args.t_out, "w"), indent=2)

    print(f"\n=== arm 3 weights ===")
    for d in sorted(domains, key=lambda x: -w[domains.index(x)]):
        print(f"  {d:14} {w[domains.index(d)]*100:6.2f}%   "
              f"(natural {natural[domains.index(d)]*100:5.2f}%)")
    print(f"\npredicted loss {pred[best]:.4f} vs natural {nat_pred:.4f}")
    print(f"aggregate law r2 = {f['r2']:.3f}")
    if f["r2"] < 0.5:
        print("WARNING: aggregate law fits poorly; its chosen weights are not trustworthy "
              "and arm 3 should be reported as such")
    print(f"-> {args.out}  and  {args.t_out} ({len(tmat)} signature rows for arm 5)")


if __name__ == "__main__":
    main()
