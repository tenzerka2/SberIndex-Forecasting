"""Time-safe feature groups for structural-change early warning.

All features for (series i, month t) are computed from arrays truncated at t (loops over t), so
per-series scales, medians and changes can never use later months. Feature groups:

  base      detectors on the total series (jump, CUSUM, Page-Hinkley, BOCPD, PELT) + state
  category  municipal monthly categories (5): panel-relative acceleration, share changes,
            cross-category dispersion, category volatility, category jump scores.
            Only for the 1,909 municipalities whose category series can be linked by name;
            NaN for the 107 homonym series.
  national  national monthly SberIndex series (total + 4 types): YoY, its acceleration, SA index
            momentum. Identical for all municipalities in a month.
  weekly    national weekly category YoY aggregated to complete months (weeks ending <= t):
            level, acceleration, cross-category dispersion, intra-month momentum.
            Identical for all municipalities in a month; available from 2023-11.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import early_warning as EW
from .data import Panel, find, load_panel, national_monthly, weekly_monthly

CATEGORIES = ["Здоровье", "Маркетплейсы", "Общественное питание", "Продовольствие", "Транспорт"]


def category_matrices(panel: Panel) -> dict[str, np.ndarray]:
    """Log category values aligned to the rows of the total panel (NaN where not linkable)."""
    key = panel.meta["mo"].where(~panel.meta["homonym"])
    out = {}
    for c in CATEGORIES:
        p = load_panel(c)
        ok = ~p.meta["homonym"].to_numpy()
        frame = pd.DataFrame(np.log(p.values[ok]), index=p.meta.loc[ok, "mo"].to_numpy())
        out[c] = frame.reindex(key.to_numpy()).to_numpy()
    return out


def _sig(x: np.ndarray) -> np.ndarray:
    """Robust scale of month-to-month changes, NaN-safe, per row."""
    dd = np.diff(x, axis=1)
    med = np.nanmedian(dd, axis=1, keepdims=True)
    return np.maximum(1.4826 * np.nanmedian(np.abs(dd - med), axis=1), EW.MIN_SIGMA)


def category_features(L: np.ndarray, C: dict[str, np.ndarray], t_from: int = 4) -> dict[str, np.ndarray]:
    n, T = L.shape
    names = ["cat_acc_max", "cat_acc_mean", "cat_jump_max", "cat_disp", "cat_vol", "cat_share_chg1_max",
             "cat_share_chg3_max", "cat_cover_chg1", "cat_n_jumping"]
    names += [f"cat{j}_{k}" for j in range(len(C)) for k in ("chg", "acc", "share_chg3")]
    F = {k: np.full((n, T), np.nan) for k in names}
    cats = list(C)
    for t in range(t_from, T):
        Lt = L[:, : t + 1]
        chg, acc, jump, vol, sh1, sh3 = [], [], [], [], [], []
        cover = np.log(np.nansum(np.exp(np.stack([C[c][:, : t + 1] for c in cats])), axis=0)) - Lt
        for j, c in enumerate(cats):
            x = C[c][:, : t + 1]
            d = x - np.nanmedian(x, axis=0)  # panel-relative category level (month-wise median)
            s = _sig(d)
            c1 = (d[:, t] - d[:, t - 1]) / s
            c0 = (d[:, t - 1] - d[:, t - 2]) / s
            a = c1 - c0
            j3 = np.abs(np.nanmedian(d[:, t - 1 : t + 1], axis=1) - np.nanmedian(d[:, t - 4 : t - 1], axis=1)) / s
            share = x - Lt
            s1, s3 = share[:, t] - share[:, t - 1], share[:, t] - share[:, t - 3]
            v = np.nanstd(np.diff(d[:, -4:], axis=1), axis=1) / s
            chg.append(c1); acc.append(a); jump.append(j3); vol.append(v); sh1.append(s1); sh3.append(s3)
            F[f"cat{j}_chg"][:, t], F[f"cat{j}_acc"][:, t], F[f"cat{j}_share_chg3"][:, t] = c1, a, s3
        chg, acc, jump = np.stack(chg), np.stack(acc), np.stack(jump)
        F["cat_acc_max"][:, t] = np.nanmax(np.abs(acc), axis=0)
        F["cat_acc_mean"][:, t] = np.nanmean(acc, axis=0)
        F["cat_jump_max"][:, t] = np.nanmax(jump, axis=0)
        F["cat_n_jumping"][:, t] = np.nansum(jump >= 3, axis=0).astype(float)
        F["cat_disp"][:, t] = np.nanstd(chg, axis=0)
        F["cat_vol"][:, t] = np.nanmean(np.stack(vol), axis=0)
        F["cat_share_chg1_max"][:, t] = np.nanmax(np.abs(np.stack(sh1)), axis=0)
        F["cat_share_chg3_max"][:, t] = np.nanmax(np.abs(np.stack(sh3)), axis=0)
        F["cat_cover_chg1"][:, t] = cover[:, t] - cover[:, t - 1]
    mask = np.isnan(C[cats[0]][:, 0])
    for k in F:
        F[k][mask] = np.nan
    return F


def _broadcast(series_by_month: pd.DataFrame, periods: pd.DatetimeIndex, n: int) -> dict[str, np.ndarray]:
    """Month-level (identical across municipalities) features -> (n x T) arrays, month t gets the
    value dated t (callers guarantee the frame only holds data available at the end of month t)."""
    s = series_by_month.reindex(periods)
    return {c: np.tile(s[c].to_numpy(dtype=float), (n, 1)) for c in s.columns}


def national_features(periods: pd.DatetimeIndex, n: int) -> dict[str, np.ndarray]:
    raw = pd.read_csv(find("consumer-spending-growth*.csv"), sep=";")
    raw["period"] = pd.to_datetime(raw["period"])
    yoy = raw[raw["value_type"].eq("Номинальное")].pivot(index="period", columns="type", values="value")
    yoy.columns = [f"nat_yoy_{c}" for c in yoy.columns]
    nat = national_monthly()
    f = pd.DataFrame(index=nat.index)
    for c in yoy.columns:
        f[c] = yoy[c]
        f[c + "_acc"] = yoy[c] - yoy[c].shift(1)
        f[c + "_acc3"] = yoy[c] - yoy[c].shift(3)
    f["nat_sa_mom1"] = np.log(nat["nat_index_sa"]).diff()
    f["nat_sa_mom3"] = np.log(nat["nat_index_sa"]).diff(3)
    f["nat_real_minus_nominal"] = nat["nat_yoy_real"] - nat["nat_yoy_nominal"]
    return _broadcast(f, periods, n)


def weekly_features(periods: pd.DatetimeIndex, n: int) -> dict[str, np.ndarray]:
    w = weekly_monthly()
    cats = [c for c in w.columns if c.startswith("wk_")]
    f = pd.DataFrame(index=w.index)
    f["w_mean"], f["w_disp"], f["w_intra_mom"] = w["w_mean"], w["w_disp"], w["w_last_minus_mean"]
    f["w_mean_acc"] = w["w_mean"].diff()
    f["w_disp_chg"] = w["w_disp"].diff()
    acc = w[cats].diff()
    f["w_acc_max"] = acc.abs().max(axis=1)
    f["w_acc_disp"] = acc.std(axis=1)
    for c in cats:
        f[c] = w[c]
        f[c + "_acc"] = acc[c]
    return _broadcast(f, periods, n)


def build_groups(panel: Panel, with_pelt: bool = True) -> dict[str, dict[str, np.ndarray]]:
    L = panel.logs
    n = L.shape[0]
    scores = {
        "jump_raw": EW.score_jump(L, seasonal=False),
        "cusum": EW.score_cusum(L),
        "page_hinkley": EW.score_page_hinkley(L),
        "bocpd": EW.score_bocpd(L),
    }
    if with_pelt:
        scores["pelt"] = EW.score_pelt(L)
    base = EW.feature_tensor(L, scores)
    return {
        "base": base,
        "category": category_features(L, category_matrices(panel)),
        "national": national_features(panel.periods, n),
        "weekly": weekly_features(panel.periods, n),
    }
