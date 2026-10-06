"""Forecasting models.

Every model is a function ``model(hist, horizons, ctx) -> {h: log_forecast}`` where ``hist`` is
the (n_series x (T+1)) matrix of LOG values truncated at the forecast origin T. A model never
receives anything observed after the origin, so time safety is structural: learned models build
their own training pairs from ``hist`` and can only use targets dated <= T.

``ctx`` carries origin-truncated exogenous data (national series) and the origin timestamp.

Notation: f_t = cross-sectional median of log values (common factor), d_it = log y_it - f_t.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

Forecast = dict[int, np.ndarray]


# ---------------------------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------------------------
def factor(hist: np.ndarray, kind: str = "median") -> np.ndarray:
    if kind == "median":
        return np.median(hist, axis=0)
    if kind == "trim":  # 20% trimmed mean per month
        s = np.sort(hist, axis=0)
        k = int(0.2 * s.shape[0])
        return s[k : s.shape[0] - k].mean(axis=0)
    if kind == "mean":
        return hist.mean(axis=0)
    raise ValueError(kind)


def mean_yoy(x: np.ndarray, k: int) -> np.ndarray:
    """Mean of the last k available YoY log growth rates (k truncated to what is observable)."""
    T = x.shape[-1] - 1
    k = max(1, min(k, T - 11))
    yoy = x[..., T - k + 1 : T + 1] - x[..., T - k - 11 : T - 11]
    return yoy.mean(axis=-1)


def factor_path(hist: np.ndarray, horizons, k: int = 6, kind: str = "median") -> dict[int, float]:
    """Common factor at T+h: same month last year advanced by mean factor YoY of last k months."""
    f = factor(hist, kind)
    T = hist.shape[1] - 1
    g = mean_yoy(f, k)
    return {h: f[T + h - 12] + g for h in horizons}


def to_level_blend(parts: list[Forecast], weights) -> Forecast:
    """Weighted average in RUB space (as the original V2 ensemble), returned in logs."""
    out = {}
    for h in parts[0]:
        w = weights[h] if isinstance(weights, dict) else weights
        out[h] = np.log(sum(wi * np.exp(p[h]) for wi, p in zip(w, parts)))
    return out


# ---------------------------------------------------------------------------------------------
# Baselines and V2 components
# ---------------------------------------------------------------------------------------------
def seasonal_naive(hist, horizons, ctx=None) -> Forecast:
    T = hist.shape[1] - 1
    return {h: hist[:, T + h - 12] for h in horizons}


def seasonal_growth(k: int):
    def model(hist, horizons, ctx=None) -> Forecast:
        T = hist.shape[1] - 1
        g = mean_yoy(hist, k)
        return {h: hist[:, T + h - 12] + g for h in horizons}

    model.__name__ = f"seasonal_growth{k}"
    return model


local_sng2 = seasonal_growth(2)


def panel_factor(k: int = 6, kind: str = "median"):
    def model(hist, horizons, ctx=None) -> Forecast:
        T = hist.shape[1] - 1
        f = factor(hist, kind)
        fp = factor_path(hist, horizons, k, kind)
        dev = hist[:, T] - f[T]
        return {h: dev + fp[h] for h in horizons}

    model.__name__ = f"panel_factor{k}"
    return model


panel_factor6 = panel_factor(6)


def v2_ensemble(hist, horizons, ctx=None) -> Forecast:
    return to_level_blend([local_sng2(hist, horizons), panel_factor6(hist, horizons)], [0.5, 0.5])


# ---------------------------------------------------------------------------------------------
# Time-safe extensions. All of them only use pairs whose target month is <= the origin T.
# ---------------------------------------------------------------------------------------------
def _idio(err: np.ndarray) -> np.ndarray:
    """Remove the cross-sectional median (common shock) from a vector of log errors."""
    return err - np.median(err)


def ctx_at(ctx: dict | None, T: int, o: int) -> dict | None:
    """Context as it was at an earlier origin o (< T): exogenous data truncated at that month."""
    if ctx is None:
        return None
    import pandas as pd

    origin = ctx["origin"] - pd.DateOffset(months=T - o)
    out = {"origin": origin}
    for k, v in ctx.items():
        if k != "origin" and hasattr(v, "index"):
            out[k] = v[v.index <= origin]
    return out


def _past_forecasts(base, hist: np.ndarray, first_origin: int, horizons, ctx=None) -> dict[int, Forecast]:
    T = hist.shape[1] - 1
    return {o: base(hist[:, : o + 1], [h for h in horizons if o + h <= T], ctx_at(ctx, T, o))
            for o in range(first_origin, T)}


def error_feedback(base=None, lags: int = 2, first_origin: int = 13, horizons=(1, 2, 3)):
    """Base forecast corrected by the series' own recent idiosyncratic one-step errors.

    The one-step error for month t is log y_t - base forecast of t made at t-1; it is known at t.
    Coefficients b_j in  e_idio(o -> o+h) = sum_j b_j * e1_idio(o - j)  are fitted by least squares,
    pooled over horizons, on all pairs with o + h <= T. Without enough history b = 0 (pure base).
    Motivation: idiosyncratic errors of adjacent origins are negatively autocorrelated (-0.08..-0.20),
    i.e. the base over-trusts the latest noisy observation.
    """
    base = base or v2_ensemble

    def model(hist, hs, ctx=None) -> Forecast:
        T = hist.shape[1] - 1
        out = base(hist, hs, ctx)
        past = _past_forecasts(base, hist, first_origin, horizons, ctx)
        e1 = {o + 1: _idio(hist[:, o + 1] - f[1]) for o, f in past.items() if 1 in f}
        X, y = [], []
        for o, f in past.items():
            if not all((o - j) in e1 for j in range(lags)):
                continue
            for h, fh in f.items():
                X.append(np.column_stack([e1[o - j] for j in range(lags)]))
                y.append(_idio(hist[:, o + h] - fh))
        if X and all((T - j) in e1 for j in range(lags)):
            X, y = np.vstack(X), np.concatenate(y)
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            corr = np.column_stack([e1[T - j] for j in range(lags)]) @ b
            out = {h: out[h] + corr for h in hs}
        return out

    model.__name__ = "error_feedback"
    return model


def blend_components(local_growth, factor_growth, w_local: float = 0.5):
    """V2 structure with pluggable common-growth estimators (functions of (f, T) -> scalar)."""

    def model(hist, hs, ctx=None) -> Forecast:
        T = hist.shape[1] - 1
        f = factor(hist)
        dev = hist[:, T] - f[T]
        u = mean_yoy(hist, 2) - mean_yoy(f, 2)  # idiosyncratic YoY of the last two months
        gl, gf = local_growth(f, T, ctx), factor_growth(f, T, ctx)
        out = {}
        for h in hs:
            local = hist[:, T + h - 12] + u + gl
            panel = dev + f[T + h - 12] + gf
            out[h] = np.log(w_local * np.exp(local) + (1 - w_local) * np.exp(panel))
        return out

    return model


def g_window(k: int):
    return lambda f, T, ctx: mean_yoy(f[: T + 1], k)


def g_national(k: int = 2):
    """Common YoY from the national SberIndex spending series (only months <= origin)."""

    def g(f, T, ctx):
        nat = ctx["national"]
        s = np.log(nat["nat_spend"])
        yoy = (s - s.shift(12)).dropna()
        yoy = yoy[yoy.index <= ctx["origin"]]
        return float(yoy.iloc[-k:].mean())

    return g


def g_weekly(k: int = 2):
    """Common YoY from weekly national category data (mean over categories of the monthly mean of
    weekly YoY %, last k complete months <= origin). Ablation only; not used in the finals."""

    def g(f, T, ctx):
        w = ctx["weekly"]
        w = w[w.index <= ctx["origin"]]["w_mean"].dropna()
        return float(np.log1p(w.iloc[-k:] / 100).mean())

    return g


def g_hedge(*estimators):
    """Equal-weight average of several common-growth estimators (no selection, no fitted weights)."""
    return lambda f, T, ctx: float(np.mean([e(f, T, ctx) for e in estimators]))


def v3_hedge(hist, horizons, ctx=None) -> Forecast:
    """V2 local part; panel part advanced by the hedge {municipal 6m, municipal 2m, national 2m}."""
    gf = g_hedge(g_window(6), g_window(2), g_national(2))
    return blend_components(g_window(2), gf)(hist, horizons, ctx)


def g_adaptive(cands=(1, 2, 3, 6), last: int = 6, first_origin: int = 12):
    """Pick the common-growth window with the lowest recent error on already-observed targets."""

    def g(f, T, ctx):
        f = f[: T + 1]
        errs = {k: [] for k in cands}
        for o in range(first_origin, T):
            for h in (1, 2, 3):
                if o + h <= T:
                    for k in cands:
                        errs[k].append(abs(f[o + h] - f[o + h - 12] - mean_yoy(f[: o + 1], k)))
        if not errs[cands[0]]:
            return 0.5 * (mean_yoy(f, 2) + mean_yoy(f, 6))
        best = min(cands, key=lambda k: np.mean(errs[k][-last:]))
        return mean_yoy(f, best)

    return g


def horizon_weights(grid=np.linspace(0, 1, 21), shrink: float = 0.5, first_origin: int = 13):
    """Per-horizon weight between local_sng2 and panel_factor6, chosen on past origins only."""

    def model(hist, hs, ctx=None) -> Forecast:
        T = hist.shape[1] - 1
        a, b = local_sng2(hist, hs), panel_factor6(hist, hs)
        past = {o: (local_sng2(hist[:, : o + 1], [1, 2, 3]), panel_factor6(hist[:, : o + 1], [1, 2, 3]))
                for o in range(first_origin, T)}
        out = {}
        for h in hs:
            pairs = [(np.exp(pa[h]), np.exp(pb[h]), np.exp(hist[:, o + h])) for o, (pa, pb) in past.items() if o + h <= T]
            w = 0.5
            if pairs:
                A = np.concatenate([p[0] for p in pairs]); B = np.concatenate([p[1] for p in pairs])
                Y = np.concatenate([p[2] for p in pairs])
                w_hat = grid[np.argmin([np.abs(Y - g * A - (1 - g) * B).mean() for g in grid])]
                w = shrink * 0.5 + (1 - shrink) * w_hat
            out[h] = np.log(w * np.exp(a[h]) + (1 - w) * np.exp(b[h]))
        return out

    return model


def lowrank_factor(rank: int = 4, k: int = 2):
    """Robust common-factor decomposition: d = a_i + Lambda_i F_t + e_it via SVD of history.

    Factors are advanced with a seasonal random walk with drift (mean of last k YoY changes); the
    common level uses the median factor with the same k-window momentum.
    """

    def model(hist, hs, ctx=None) -> Forecast:
        T = hist.shape[1] - 1
        f = factor(hist)
        D = hist - f
        a = D.mean(axis=1, keepdims=True)
        U, s, Vt = np.linalg.svd(D - a, full_matrices=False)
        lam, F = U[:, :rank] * s[:rank], Vt[:rank]
        E = D - a - lam @ F
        g = mean_yoy(f, k)
        e_last = E[:, T - k + 1 : T + 1].mean(axis=1)
        out = {}
        for h in hs:
            dF = np.mean([F[:, T - j] - F[:, T - j - 12] for j in range(k)], axis=0)
            out[h] = a[:, 0] + lam @ (F[:, T + h - 12] + dF) + e_last + f[T + h - 12] + g
        return out

    return model


# --- pooled learners -------------------------------------------------------------------------
def pooled_features(hist: np.ndarray, o: int, h: int, f: np.ndarray | None = None) -> np.ndarray:
    f = factor(hist[:, : o + 1]) if f is None else f
    D = hist[:, : o + 1] - f[: o + 1]
    yoy = hist[:, o] - hist[:, o - 12]
    cols = [
        np.full(len(D), h),
        np.full(len(D), (o + h) % 12),
        yoy, hist[:, o - 1] - hist[:, o - 13] if o >= 13 else yoy,
        D[:, o], D[:, o - 1], D[:, o - 2], D[:, o + h - 12], D[:, o - 12],
        D[:, o + h - 12] - D[:, o - 12],
        np.std(np.diff(D, axis=1), axis=1),
        np.full(len(D), mean_yoy(f[: o + 1], 2)), np.full(len(D), mean_yoy(f[: o + 1], 6)),
    ]
    return np.column_stack(cols)


def pooled_gbm(kind: str = "lgbm", residual: bool = True, min_origin: int = 13, seed: int = 42):
    """Global GBM over all series, trained on pairs with o + h <= T.

    residual=True  -> learns the log error of v2_ensemble (residual learning on top of V2);
    residual=False -> learns log YoY growth of the target month directly.
    """

    def make():
        if kind == "lgbm":
            from lightgbm import LGBMRegressor

            return LGBMRegressor(objective="l1", n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=200, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, verbosity=-1,
                                 n_jobs=2)
        from catboost import CatBoostRegressor

        return CatBoostRegressor(loss_function="MAE", iterations=400, learning_rate=0.05, depth=4,
                                 l2_leaf_reg=10, random_seed=seed, verbose=False, thread_count=2)

    def model(hist, hs, ctx=None) -> Forecast:
        T = hist.shape[1] - 1
        base_T = v2_ensemble(hist, hs)
        X, y, w = [], [], []
        for o in range(min_origin, T):
            ho = hist[:, : o + 1]
            fo = factor(ho)
            b = v2_ensemble(ho, [h for h in (1, 2, 3) if o + h <= T]) if residual else None
            for h in (1, 2, 3):
                if o + h > T:
                    continue
                X.append(pooled_features(hist, o, h, fo))
                tgt = hist[:, o + h] - (b[h] if residual else hist[:, o + h - 12])
                y.append(tgt)
                w.append(np.exp(hist[:, o + h]))
        if not X:
            return base_T
        X, y, w = np.vstack(X), np.concatenate(y), np.concatenate(w)
        m = make().fit(X, y, sample_weight=w / w.mean())
        f = factor(hist)
        out = {}
        for h in hs:
            pred = m.predict(pooled_features(hist, T, h, f))
            out[h] = base_T[h] + pred if residual else hist[:, T + h - 12] + pred
        return out

    return model


MODELS: dict[str, Callable] = {
    "seasonal_naive": seasonal_naive,
    "seasonal_growth1": seasonal_growth(1),
    "local_sng2": local_sng2,
    "panel_factor6": panel_factor6,
    "v2_ensemble": v2_ensemble,
}
