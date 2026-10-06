"""Rolling-origin backtest engine shared by all benchmarks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel, month_index

EXACT_ORIGINS = pd.to_datetime(["2024-06-01", "2024-07-01", "2024-08-01", "2024-09-01"])
# Every origin for which a 3-month-ahead target exists and YoY growth of the last 2 months is
# observable (needs T-13 >= 2023-01). Horizons beyond 2024-12 are dropped.
EXTENDED_ORIGINS = pd.date_range("2024-02-01", "2024-11-01", freq="MS")
HORIZONS = [1, 2, 3]


def run(panel: Panel, models: dict, origins=EXACT_ORIGINS, horizons=HORIZONS, ctx_fn=None) -> pd.DataFrame:
    L = panel.logs
    last = L.shape[1] - 1
    frames = []
    for origin in origins:
        T = month_index(origin)
        hs = [h for h in horizons if T + h <= last]
        hist = L[:, : T + 1].copy()  # the only thing a model ever sees
        hist.setflags(write=False)
        ctx = ctx_fn(origin) if ctx_fn else {"origin": origin}
        preds = {name: m(hist, hs, ctx) for name, m in models.items()}
        for h in hs:
            block = {
                "series": np.arange(L.shape[0]),
                "origin": origin,
                "h": h,
                "target_date": panel.periods[T + h],
                "y": np.exp(L[:, T + h]),
                "y_base": np.exp(L[:, T + h - 12]),
            }
            for name in models:
                block[name] = np.exp(preds[name][h])
            frames.append(pd.DataFrame(block))
    return pd.concat(frames, ignore_index=True)


def r2(a, b) -> float:
    a = np.asarray(a)
    return float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))


def metrics(df: pd.DataFrame, col: str) -> dict:
    y, p = df["y"].to_numpy(), df[col].to_numpy()
    g, gh = np.log(y / df["y_base"]), np.log(p / df["y_base"])
    ae = np.abs(y - p)
    return {
        "n": len(df),
        "MAE": float(ae.mean()),
        "R2_level": r2(y, p),
        "R2_growth": r2(g, gh),
        "wMAPE_pct": float(100 * ae.sum() / np.abs(y).sum()),
    }


def table(df: pd.DataFrame, cols, by=None) -> pd.DataFrame:
    rows = []
    groups = [((), df)] if by is None else df.groupby(by)
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        for c in cols:
            rows.append({**dict(zip([] if by is None else ([by] if isinstance(by, str) else by), key)),
                         "model": c, **metrics(g, c)})
    return pd.DataFrame(rows)


def block_bootstrap_diff(df: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired comparison of MAE(a) - MAE(b) (negative = a better).

    * percentile bootstrap by series and by target month (all series share the macro shock of a
      month, so the month is the dominant unit of dependence). With only 6-8 target months the
      month bootstrap is anti-conservative; treat its CI as optimistic.
    * exact one-sided sign test over target months (H0: a and b equally likely to win a month),
      the most defensible statement with so few clusters.
    """
    from scipy.stats import binomtest

    rng = np.random.default_rng(seed)
    d = (df[a] - df["y"]).abs() - (df[b] - df["y"]).abs()
    out = {"diff": float(d.mean())}
    for unit in ["series", "target_date"]:
        per = d.groupby(df[unit]).agg(["sum", "count"])
        s, c = per["sum"].to_numpy(), per["count"].to_numpy()
        idx = rng.integers(0, len(s), size=(n_boot, len(s)))
        bs = s[idx].sum(1) / c[idx].sum(1)
        out[f"ci95_by_{unit}"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        # share of bootstrap replicates on the other side of zero (one-sided bootstrap p-value)
        out[f"p_boot_by_{unit}"] = float((bs >= 0).mean()) if out["diff"] < 0 else float((bs <= 0).mean())
    per_month = d.groupby(df["target_date"]).mean()
    wins, n = int((per_month < 0).sum()), int(len(per_month))
    out["months_better"], out["months_total"] = wins, n
    out["sign_test_p_one_sided"] = float(binomtest(wins, n, 0.5, alternative="greater").pvalue) if out["diff"] < 0 \
        else float(binomtest(n - wins, n, 0.5, alternative="greater").pvalue)
    return out
