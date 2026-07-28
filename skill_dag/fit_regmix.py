"""
Arm 2: pick FIXED domain weights via regression on the proxy fleet. CPU only.

The doc: "Select optimal domain weight based on the regression model and keep those weights
throughout training. Use incomplete training runs of various 50-100M sized models and
extrapolation with the power laws to fit a LightGBM regression for val loss as a function
of domain weights. Then minimize that function."

Pipeline:
  fleet.jsonl -> power-law extrapolate each run to convergence (extrapolate.py)
              -> power-law extrapolate across model size to the 1B target
              -> LightGBM: weight vector -> predicted target loss
              -> minimise by search WITHIN the Dirichlet support the fleet sampled

Minimising inside the support is deliberate. A boosted-tree regressor fitted on mixtures
drawn near the natural mix says nothing reliable about a corner of the simplex it never
saw, and its predictions there are flat extrapolations that can look artificially good.
Candidates are therefore drawn from the SAME Dirichlet the fleet used.

LightGBM is used when installed; otherwise this falls back to sklearn's
GradientBoostingRegressor and then to ridge regression, reporting which ran. The fallback
is noted in the output because arm 2 vs arm 3 is a comparison of regressor family, so
which regressor actually ran is part of the result.

Usage:
  python fit_regmix.py --fleet fleet --out weights_arm2.json
"""
import argparse, json, os

import numpy as np

from extrapolate import converged_loss, extrapolate_to_size, select_in_support

TARGET_PARAMS = 1.177e9          # OLMo-1B total parameters


def load_fleet(path):
    runs = []
    with open(os.path.join(path, "fleet.jsonl")) as f:
        for line in f:
            if line.strip():
                runs.append(json.loads(line))
    if not runs:
        raise SystemExit(f"no runs in {path}/fleet.jsonl")
    return runs


def build_dataset(runs, key, target_params, verbose=True):
    """-> (X weights, y predicted loss at target size, domains, diagnostics)"""
    domains = sorted(runs[0]["weights"])
    by_mix = {}
    diag = {"curve_fits_ok": 0, "curve_fits_fallback": 0,
            "size_fits_ok": 0, "size_fits_fallback": 0}
    for r in runs:
        E, info = converged_loss(r["curve"], key)
        if E is None:
            continue
        diag["curve_fits_ok" if info["ok"] else "curve_fits_fallback"] += 1
        mi = r["mixture_index"]
        by_mix.setdefault(mi, {"weights": r["weights"], "sizes": {}})
        by_mix[mi]["sizes"][float(r["cost"]["params"])] = E

    X, y, kept = [], [], []
    for mi, rec in sorted(by_mix.items()):
        pred, info = extrapolate_to_size(rec["sizes"], target_params)
        if pred is None:
            continue
        diag["size_fits_ok" if info["ok"] else "size_fits_fallback"] += 1
        X.append([rec["weights"][d] for d in domains])
        y.append(pred)
        kept.append(mi)
    if verbose:
        print(f"  curve fits: {diag['curve_fits_ok']} trusted, "
              f"{diag['curve_fits_fallback']} fell back to last value")
        print(f"  size fits:  {diag['size_fits_ok']} trusted, "
              f"{diag['size_fits_fallback']} fell back to largest size")
    return np.array(X), np.array(y), domains, diag, kept


def fit_regressor(X, y):
    """LightGBM -> sklearn GBM -> ridge. Returns (predict_fn, name)."""
    try:
        import lightgbm as lgb
        ds = lgb.Dataset(X, label=y)
        params = {"objective": "regression", "verbosity": -1,
                  "num_leaves": 7, "learning_rate": 0.05,
                  "min_data_in_leaf": 3, "feature_fraction": 0.9}
        booster = lgb.train(params, ds, num_boost_round=400)
        return (lambda Z: booster.predict(Z)), "lightgbm"
    except ImportError:
        pass
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        m = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3)
        m.fit(X, y)
        return (lambda Z: m.predict(Z)), "sklearn_gbm"
    except ImportError:
        pass
    # ridge on [weights, squares] -- crude but never silently unavailable
    def feats(Z):
        return np.hstack([Z, Z ** 2, np.ones((len(Z), 1))])
    A = feats(X)
    coef = np.linalg.lstsq(A.T @ A + 1e-6 * np.eye(A.shape[1]), A.T @ y, rcond=None)[0]
    return (lambda Z: feats(Z) @ coef), "ridge_quadratic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", required=True, help="directory containing fleet.jsonl")
    ap.add_argument("--out", default="weights_arm2.json")
    ap.add_argument("--key", default="_mean",
                    help="validation target: '_mean' or a specific domain")
    ap.add_argument("--target-params", type=float, default=TARGET_PARAMS)
    ap.add_argument("--candidates", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--allow-extrapolation", action="store_true",
                    help="permit predictions below every observed loss (off by default)")
    args = ap.parse_args()

    runs = load_fleet(args.fleet)
    print(f"fleet: {len(runs)} runs, target={args.key}")
    X, y, domains, diag, kept = build_dataset(runs, args.key, args.target_params)
    if len(X) < 4:
        raise SystemExit(f"only {len(X)} usable mixtures; need more fleet runs")
    print(f"  usable mixtures: {len(X)}   loss range {y.min():.4f}-{y.max():.4f}")

    predict, name = fit_regressor(X, y)
    resid = y - predict(X)
    print(f"  regressor: {name}   in-sample RMSE {np.sqrt((resid**2).mean()):.5f}")

    # candidates from the SAME Dirichlet the fleet sampled -> stay in-distribution
    mixmeta = json.load(open(os.path.join(args.fleet, "mixtures.json")))
    alpha_scale = mixmeta.get("alpha_scale", 20.0)
    natural = np.array([mixmeta["mixtures"][0][d] for d in domains], dtype=float)
    natural = natural / natural.sum()
    rng = np.random.default_rng(args.seed)
    C = rng.dirichlet(natural * alpha_scale, size=args.candidates)
    C = np.vstack([C, natural[None, :]])          # natural always a candidate
    pred = predict(C)
    best, sel = select_in_support(C, X, y, pred,
                                 allow_extrapolation=args.allow_extrapolation)
    if best is None:
        best = int(np.argmin(y))                  # fall back to the observed best mixture
        w = X[best] / X[best].sum()
        sel["selection"] = "fell back to the best OBSERVED fleet mixture"
        pred_best = float(y[best])
    else:
        w = C[best] / C[best].sum()
        pred_best = float(pred[best])
    print(f"  candidate search: {sel['in_box']}/{sel['candidates']} inside the fleet's "
          f"per-domain range, {sel['eligible']} eligible")
    if sel["raw_best_below_floor"]:
        print(f"  NOTE: unconstrained argmin predicted {sel['raw_best_prediction']:.4f}, "
              f"below the observed floor {sel['observed_loss_floor']:.4f} -- rejected as "
              f"regressor extrapolation")

    nat_pred = float(pred[-1])
    out = {
        "arm": 2, "method": f"power-law + {name}", "regressor": name,
        "weights": {d: float(w[i]) for i, d in enumerate(domains)},
        "predicted_loss": pred_best,
        "predicted_loss_natural": nat_pred,
        "predicted_improvement_vs_natural": nat_pred - pred_best,
        "n_fleet_runs": len(runs), "n_usable_mixtures": len(X),
        "in_sample_rmse": float(np.sqrt((resid ** 2).mean())),
        "extrapolation_diagnostics": diag,
        "candidates_searched": len(C), "alpha_scale": alpha_scale,
        "support_selection": sel,
        "support_note": ("candidates drawn from the fleet's own Dirichlet; the regressor is "
                         "not queried outside the region it was fitted on"),
        "config": vars(args),
    }
    json.dump(out, open(args.out, "w"), indent=2)

    print(f"\n=== arm 2 weights ({name}) ===")
    for d in sorted(domains, key=lambda x: -out["weights"][x]):
        nat = mixmeta["mixtures"][0][d]
        print(f"  {d:14} {out['weights'][d]*100:6.2f}%   (natural {nat*100:5.2f}%)")
    print(f"\npredicted loss {out['predicted_loss']:.4f} vs natural {nat_pred:.4f} "
          f"(improvement {out['predicted_improvement_vs_natural']:.4f})")
    if diag["curve_fits_fallback"] > diag["curve_fits_ok"]:
        print("WARNING: most curve fits fell back to last-observed -- the fleet runs are "
              "probably too short to show a trend (review item 5)")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
