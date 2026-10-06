"""Time-safe two-part conformal prediction intervals on top of any point forecast.

Calibration set for origin T, horizon h: earlier forecasts of the same model whose target month is
<= T (their errors are observed at T). The log error is split the same way as the forecast error:

  common part   c_{o,h} = cross-sectional median log error of that (origin, h) -> one macro shock
  idiosyncratic z = (log error - c) / s_i, s_i = robust per-series scale from data <= forecast origin

Why two parts: a single pooled conformal score mixes regimes. Origins 2024-02..04 carry a common
bias of about +2 sigma (the Q1 level anomaly), which makes asymmetric pooled intervals shift upward
(61% coverage of a nominal 80% on the exact window) and symmetric pooled ones inflate (97%).
Idiosyncratic scores are stable across regimes; the common shock is handled by a separate band with
a robust scale (1.4826 * median |c|), combined as independent normal-like components. Both parts
use the same calibration window: pairs whose target month lies in the last 4 months before T.

Half-width in logs:  sqrt((q_lv(|z|) * s_i)^2 + (z_lv * sigma_common * sqrt(h))^2)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .early_warning import rel_level, robust_sigma

IDIO_WINDOW_MONTHS = 4


def series_scale(L: np.ndarray, T: int) -> np.ndarray:
    return robust_sigma(rel_level(L[:, : T + 1]))


def _prepare(df: pd.DataFrame, col: str, L: np.ndarray, periods: pd.DatetimeIndex) -> pd.DataFrame:
    out = df.copy()
    idx = {p: i for i, p in enumerate(periods)}
    T_of = out["origin"].map(idx)
    scales = {T: series_scale(L, T) for T in T_of.unique()}
    out["_s"] = [scales[T][i] for T, i in zip(T_of, out["series"])]
    lerr = np.log(out["y"] / out[col])
    out["_c"] = lerr.groupby([out["origin"], out["h"]]).transform("median")
    out["_z"] = (lerr - out["_c"]) / out["_s"]
    return out


def calibrate(calib: pd.DataFrame, origin: pd.Timestamp, h: int, levels=(0.8, 0.9)) -> dict | None:
    """Half-width ingredients for (origin, h) from a prepared calibration frame.

    Uses only pairs with target_date <= origin, horizon <= h, target in the last 4 months.
    Returns {level: (q_idio, z_common * sigma_common * sqrt(h))} or None if too few pairs.
    """
    past = calib[(calib["target_date"] <= origin) & (calib["h"] <= h)]
    recent = past[past["target_date"] > origin - pd.DateOffset(months=IDIO_WINDOW_MONTHS)]
    if len(recent) < 1000:
        return None
    zabs = np.abs(recent["_z"]) * np.sqrt(h / recent["h"])
    common = recent.groupby(["origin", "h"])["_c"].first()
    sig_c = 1.4826 * np.median(np.abs(common.to_numpy() / np.sqrt(common.index.get_level_values("h"))))
    return {lv: (float(np.quantile(zabs, lv)), float(norm.ppf(0.5 + lv / 2) * sig_c * np.sqrt(h))) for lv in levels}


def conformal(df: pd.DataFrame, col: str, L: np.ndarray, periods: pd.DatetimeIndex,
              levels=(0.8, 0.9)) -> pd.DataFrame:
    out = _prepare(df, col, L, periods)
    for origin in sorted(out["origin"].unique()):
        cur_o = out["origin"] == origin
        for h in sorted(out.loc[cur_o, "h"].unique()):
            m = cur_o & (out["h"] == h)
            cal = calibrate(out, origin, h, levels)
            for lv in levels:
                lo, hi = f"{col}_lo{int(lv*100)}", f"{col}_hi{int(lv*100)}"
                if cal is None:
                    out.loc[m, [lo, hi]] = np.nan
                    continue
                q, c = cal[lv]
                hw = np.sqrt((q * out.loc[m, "_s"]) ** 2 + c ** 2)
                out.loc[m, lo] = out.loc[m, col] * np.exp(-hw)
                out.loc[m, hi] = out.loc[m, col] * np.exp(hw)
    return out.drop(columns=[c for c in out.columns if c.startswith("_")])


def future_intervals(calib_df: pd.DataFrame, col: str, fut: pd.DataFrame, L: np.ndarray,
                     periods: pd.DatetimeIndex, levels=(0.8, 0.9)) -> pd.DataFrame:
    """Intervals for genuine out-of-sample forecasts `fut` (series, origin, h, <col>) using the
    backtest frame `calib_df` (with outcomes) as calibration. Horizons beyond those in the
    calibration frame borrow shorter horizons scaled by sqrt(h / h') (an assumption, not tested)."""
    calib = _prepare(calib_df, col, L, periods)
    out = fut.copy()
    origin = out["origin"].iloc[0]
    T = {p: i for i, p in enumerate(periods)}[origin]
    s = series_scale(L, T)[out["series"].to_numpy()]
    for h in sorted(out["h"].unique()):
        m = (out["h"] == h).to_numpy()
        cal = calibrate(calib, origin, int(h), levels)
        for lv in levels:
            q, c = cal[lv]
            hw = np.sqrt((q * s[m]) ** 2 + c ** 2)
            out.loc[m, f"{col}_lo{int(lv*100)}"] = out.loc[m, col].to_numpy() * np.exp(-hw)
            out.loc[m, f"{col}_hi{int(lv*100)}"] = out.loc[m, col].to_numpy() * np.exp(hw)
    return out


def interval_metrics(df: pd.DataFrame, col: str, levels=(0.8, 0.9)) -> list[dict]:
    rows = []
    for lv in levels:
        lo, hi = df[f"{col}_lo{int(lv*100)}"], df[f"{col}_hi{int(lv*100)}"]
        ok = lo.notna()
        if not ok.any():
            continue
        y = df.loc[ok, "y"]; lo, hi = lo[ok], hi[ok]
        a = 1 - lv
        cover = (y >= lo) & (y <= hi)
        winkler = (hi - lo) + 2 / a * (lo - y).clip(lower=0) + 2 / a * (y - hi).clip(lower=0)
        by_month = cover.groupby(df.loc[ok, "target_date"]).mean()
        rows.append({
            "level": lv, "n": int(ok.sum()), "coverage": float(cover.mean()),
            "coverage_min_month": float(by_month.min()), "coverage_max_month": float(by_month.max()),
            "mean_width_rub": float((hi - lo).mean()), "winkler": float(winkler.mean()),
        })
    return rows
