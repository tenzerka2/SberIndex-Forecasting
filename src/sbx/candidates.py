"""Predeclared research candidates. Frozen V3 remains available unchanged.

All fitting uses observed targets only. These candidates are retrospective research;
selection on this export does not constitute an untouched final test.
"""
from __future__ import annotations

import numpy as np
from . import models as M


def robust_common(f, gain=0.35):
    """Local-level filter of observed log YoY, with bounded innovations."""
    yy = f[12:] - f[:-12]
    if not len(yy):
        raise ValueError("At least 13 months are required")
    level = float(yy[0])
    for j in range(1, len(yy)):
        history = np.diff(yy[:j + 1])
        scale = max(0.02, 1.4826 * float(np.median(np.abs(history - np.median(history)))))
        level += gain * float(np.clip(yy[j] - level, -2 * scale, 2 * scale))
    return level


def filtered_panel(hist, hs, ctx=None, common="robust", smooth=True):
    """Seasonal local estimate plus a filtered relative level and common path."""
    T = hist.shape[1] - 1
    f = M.factor(hist)
    d = hist - f
    state = d[:, 0].copy()
    for t in range(1, T + 1):
        state += 0.65 * (d[:, t] - state)
    if not smooth:
        state = d[:, -1]
    g = robust_common(f) if common == "robust" else M.mean_yoy(f, 6)
    if common == "hedge":
        g = np.mean([robust_common(f), M.mean_yoy(f, 2), M.g_national(2)(f, T, ctx)])
    local = M.local_sng2(hist, hs)
    return {h: np.log(0.5 * np.exp(local[h]) + 0.5 * np.exp(state + f[T + h - 12] + g)) for h in hs}


def filtered6(hist, hs, ctx=None):
    return filtered_panel(hist, hs, ctx, common="six")


def robust_unsmoothed(hist, hs, ctx=None):
    return filtered_panel(hist, hs, ctx, smooth=False)


def filtered_hedge(hist, hs, ctx=None):
    return filtered_panel(hist, hs, ctx, common="hedge")


def _features(hist, h):
    d = hist - M.factor(hist)
    T = hist.shape[1] - 1
    # Relative features cannot fit a national anomaly from only a handful of dates.
    a = d[:, T] - d[:, T - 1]
    b = d[:, T - 1] - d[:, T - 2]
    c = d[:, T - 2] - d[:, T - 3]
    yoy = d[:, T] - d[:, T - 12]
    prev_yoy = d[:, T - 1] - d[:, T - 13]
    seasonal = d[:, T + h - 12] - d[:, T - 12]
    X = np.column_stack([a, b, c, yoy, prev_yoy, seasonal,
                         d[:, T] - np.median(d[:, -6:], axis=1)])
    return X - np.median(X, axis=0)


def residual_ridge(base=M.v2_ensemble, rub_loss=False):
    """Small pooled residual model with shrinkage, optionally aligned to RUB MAE.

Seven features, fixed penalty and clipping. No fitting on the evaluation targets.
IRLS is a smooth approximation to absolute RUB error, not a new test set.
"""
    def model(hist, hs, ctx=None):
        T = hist.shape[1] - 1
        out = base(hist, hs, ctx)
        Xs, ys, weights, origin_ids = [], [], [], []
        for o in range(13, T):
            horizons = [h for h in (1, 2, 3) if o + h <= T]
            if not horizons:
                continue
            H = hist[:, :o + 1]
            p = base(H, horizons, M.ctx_at(ctx, T, o))
            for h in horizons:
                Xs.append(_features(H, h))
                err = hist[:, o + h] - p[h]
                ys.append(err - np.median(err))
                weights.append(np.exp(p[h]))
                origin_ids.append(o)
        if len(set(origin_ids)) < 2:
            return out
        X, y = np.vstack(Xs), np.concatenate(ys)
        scale = np.maximum(np.std(X, axis=0), 0.005)
        X = np.clip(X / scale, -8, 8)
        penalty = 0.25 * np.eye(X.shape[1])
        w = np.concatenate(weights) if rub_loss else np.ones(len(y))
        w /= w.mean()
        beta = np.zeros(X.shape[1])
        for _ in range(6 if rub_loss else 1):
            wi = w / np.maximum(np.abs(y - X @ beta), 0.01) if rub_loss else w
            wi /= wi.mean()
            beta = np.linalg.solve((X.T * wi) @ X / len(y) + penalty,
                                   X.T @ (wi * y) / len(y))
        return {h: out[h] + np.clip(np.clip(_features(hist, h) / scale, -8, 8) @ beta, -0.15, 0.15)
                for h in hs}
    return model


def research_models():
    return {
        "filtered_panel6": filtered6,
        "robust_common": robust_unsmoothed,
        "robust_filtered": filtered_panel,
        "robust_filtered_hedge": filtered_hedge,
        "ridge_relative": residual_ridge(),
        "ridge_relative_mae": residual_ridge(rub_loss=True),
        "ridge_hedge_mae": residual_ridge(base=M.v3_hedge, rub_loss=True),
    }


def national_shape(hist, hs, ctx=None, weight=0.5):
    """Borrow national seasonal SHAPE, not just its YoY growth rate.

    This specifically targets measurement anomalies in the municipality common factor.
    The recent log-level offset is estimated at the origin, never from future outcomes.
    Category transfer is reported separately because a total-spending national series may
    have a different seasonal shape from individual categories.
    """
    import pandas as pd
    if ctx is None:
        raise ValueError("National series and origin are required")
    origin = ctx["origin"]
    nat = np.log(ctx["national"].loc[:origin, "nat_spend"].dropna())
    f = M.factor(hist); d = hist - f; T = hist.shape[1] - 1
    dates = pd.date_range(end=origin, periods=hist.shape[1], freq="MS")
    aligned = nat.reindex(dates).to_numpy()
    if not np.isfinite(aligned).all():
        raise ValueError("National history must cover the municipal history")
    offset = np.median((f - aligned)[-3:])
    growth = (nat - nat.shift(12)).dropna().iloc[-2:].mean()
    local_g = M.mean_yoy(d, 2)
    out = {}
    for h in hs:
        date = origin + pd.DateOffset(months=h - 12)
        fp = float(nat.loc[date] + growth + offset)
        relative = np.log(0.5 * np.exp(d[:, T + h - 12] + local_g) + 0.5 * np.exp(d[:, T]))
        borrowed = fp + relative
        original = M.v2_ensemble(hist, [h])[h]
        out[h] = np.log(weight * np.exp(borrowed) + (1 - weight) * np.exp(original))
    return out


def national_shape_full(hist, hs, ctx=None):
    return national_shape(hist, hs, ctx, weight=1.0)


def diversified_base(hist, hs, ctx=None):
    """Fixed equal mixture of the robust municipal filter and the national-growth hedge."""
    return M.to_level_blend([filtered_panel(hist, hs, ctx), M.v3_hedge(hist, hs, ctx)], [0.5, 0.5])


def v4_models():
    return {"robust_filtered_feedback": M.error_feedback(base=filtered_panel),
            "diversified_base": diversified_base,
            "v4_diversified": M.error_feedback(base=diversified_base)}
