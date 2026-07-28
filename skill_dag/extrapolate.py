"""
Power-law extrapolation shared by arms 2 and 3. CPU only.

Both arms need the same thing from the fleet's incomplete runs: an estimate of the loss a
mixture would reach if trained to convergence at the target model size. Only the regressor
that maps weights -> loss differs (LightGBM for arm 2, the parametric data-mixing law for
arm 3), so this step lives in one place and both import it.

Two nested fits, following Data Mixing Laws:
  over tokens: L(S) = E + B / S^a      -> E is the converged loss at that model size
  over size:   E(N) = E_inf + C / N^b  -> E_inf is the converged loss at the target size

Both are fitted by least squares on a log grid over the exponent (no SciPy dependency):
for a fixed exponent the model is linear in its other two parameters, so each candidate
exponent has a closed-form solution and we simply keep the best.
"""
import math


def _linfit(xs, ys):
    """Least squares y = m*x + c. Returns (m, c, sse)."""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    if abs(den) < 1e-18:
        return 0.0, sy / n, sum((y - sy / n) ** 2 for y in ys)
    m = (n * sxy - sx * sy) / den
    c = (sy - m * sx) / n
    sse = sum((y - (m * x + c)) ** 2 for x, y in zip(xs, ys))
    return m, c, sse


def fit_power(xs, ys, exp_lo=0.02, exp_hi=2.0, steps=160):
    """Fit y = E + B / x^a. Returns (E, B, a, sse, r2).

    For a fixed a, u = x^-a makes it linear (y = B*u + E), which is solved exactly. We
    scan a on a log grid and keep the lowest SSE. Needs >= 3 points to be meaningful;
    with 2 it degenerates to a line through them and E is not identifiable, so we say so.
    """
    if len(xs) < 2:
        return (ys[0] if ys else float("nan")), 0.0, 0.0, 0.0, 0.0
    best = None
    for i in range(steps):
        a = exp_lo * (exp_hi / exp_lo) ** (i / (steps - 1))
        us = [x ** (-a) for x in xs]
        B, E, sse = _linfit(us, ys)
        if best is None or sse < best[3]:
            best = (E, B, a, sse)
    E, B, a, sse = best
    mu = sum(ys) / len(ys)
    tss = sum((y - mu) ** 2 for y in ys) or 1e-18
    return E, B, a, sse, 1.0 - sse / tss


MIN_R2 = 0.5        # below this the curve carries no usable trend
MAX_GAIN = 0.40     # extrapolating >40% below the last measurement is not credible


def converged_loss(curve, key, min_r2=MIN_R2, max_gain=MAX_GAIN):
    """Extrapolate one incomplete run's curve to S -> infinity for validation domain `key`.

    curve: [{"tokens": int, "val_loss": {domain: loss, "_mean": ...}}, ...]

    Falls back to the LAST OBSERVED value whenever the fit cannot be trusted. This matters
    more than it looks: a flat, noisy curve (the signature of a run that was too short to
    show a trend) fits happily with a tiny exponent and a huge coefficient, which drives E
    far below anything measured. Without the r^2 and gain guards, noise becomes a large
    predicted improvement and both arm 2's regressor and arm 3's law optimise toward it.
    """
    pts = [(c["tokens"], c["val_loss"][key]) for c in curve if key in c["val_loss"]]
    if not pts:
        return None, {"ok": False, "reason": "no points"}
    pts.sort()
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    last = ys[-1]
    if len(pts) < 3:
        return last, {"ok": False, "reason": f"only {len(pts)} points; used last value",
                      "last": last}
    E, B, a, sse, r2 = fit_power(xs, ys)
    if not math.isfinite(E) or not (0.0 < E <= last + 1e-9):
        return last, {"ok": False, "reason": "implausible E; used last value",
                      "E_raw": E, "last": last, "r2": r2}
    if r2 < min_r2:
        return last, {"ok": False, "reason": f"r2 {r2:.3f} < {min_r2}; no usable trend, "
                                             f"used last value", "E_raw": E, "last": last,
                      "r2": r2}
    if last > 0 and (last - E) / last > max_gain:
        return last, {"ok": False,
                      "reason": f"implied gain {(last-E)/last:.1%} > {max_gain:.0%}; "
                                f"used last value", "E_raw": E, "last": last, "r2": r2}
    return E, {"ok": True, "E": E, "B": B, "a": a, "r2": r2, "last": last,
               "gain_vs_last": last - E}


def select_in_support(C, X, y, pred, allow_extrapolation=False):
    """Pick the best candidate that the regressor can actually speak to.

    Two independent ways a weight-search goes wrong, both seen in testing:

    1. The candidate sits outside the per-domain weight range the fleet ever trained on.
       Being inside the Dirichlet simplex is NOT enough -- a Dirichlet around the natural
       mix still reaches corners no fleet run visited.
    2. The predicted loss falls below every loss ever measured. An unbounded regressor
       (quadratic ridge especially) will happily report a near-zero loss at the edge of
       its domain. That is the fit escaping, not a better mixture.

    Returns (index, info). Falls back to the best in-range candidate, or to the observed
    best mixture if nothing qualifies.
    """
    import numpy as np
    lo, hi = X.min(axis=0), X.max(axis=0)
    in_box = np.all((C >= lo - 1e-12) & (C <= hi + 1e-12), axis=1)
    floor = float(y.min())
    credible = pred >= floor if not allow_extrapolation else np.ones(len(pred), bool)
    ok = in_box & credible
    info = {"candidates": int(len(C)), "in_box": int(in_box.sum()),
            "credible": int(credible.sum()), "eligible": int(ok.sum()),
            "observed_loss_floor": floor,
            "raw_best_prediction": float(pred.min()),
            "raw_best_below_floor": bool(pred.min() < floor)}
    if ok.any():
        idx = int(np.where(ok)[0][np.argmin(pred[ok])])
        info["selection"] = "best in-support candidate"
        return idx, info
    if in_box.any():
        idx = int(np.where(in_box)[0][np.argmin(pred[in_box])])
        info["selection"] = ("no candidate predicted above the observed floor; used best "
                             "in-box candidate and flagged extrapolation")
        return idx, info
    info["selection"] = "no in-box candidate; caller should fall back to observed best"
    return None, info


def extrapolate_to_size(size_to_loss, target_params, min_sizes=2, min_r2=MIN_R2):
    """E(N) = E_inf + C / N^b across model sizes -> loss at target_params.

    size_to_loss: {param_count: converged_loss}. With fewer than `min_sizes` distinct
    sizes this cannot be fitted, so the largest available size's value is returned and
    flagged -- silently pretending to extrapolate would be worse.
    """
    items = sorted(size_to_loss.items())
    if len(items) < min_sizes:
        v = items[-1][1] if items else None
        return v, {"ok": False, "reason": f"{len(items)} size(s); used largest", "value": v}
    xs = [float(n) for n, _ in items]
    ys = [float(v) for _, v in items]
    E, B, a, sse, r2 = fit_power(xs, ys)
    largest = ys[-1]
    if not math.isfinite(E) or E <= 0:
        return largest, {"ok": False, "reason": "implausible E_inf; used largest size",
                         "value": largest, "r2": r2}
    if r2 < min_r2:
        return largest, {"ok": False,
                         "reason": f"r2 {r2:.3f} < {min_r2} across sizes; used largest",
                         "value": largest, "r2": r2}
    pred = E + B * target_params ** (-a)
    if not math.isfinite(pred) or pred <= 0:
        return largest, {"ok": False, "reason": "implausible prediction; used largest size",
                         "value": largest, "r2": r2}
    return pred, {"ok": True, "E_inf": E, "B": B, "b": a, "r2": r2,
                  "at_target": pred, "largest_observed": largest}
