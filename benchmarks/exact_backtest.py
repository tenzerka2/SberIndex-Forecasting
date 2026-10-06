from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbx import data as sbx_data  # noqa: E402

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

ORIGINS = pd.to_datetime(["2024-06-01", "2024-07-01", "2024-08-01", "2024-09-01"])
HORIZONS = [1, 2, 3]


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct series identity from raw (mo, category) row runs (see src/sbx/data.py).

    The export is a shuffle of per-(municipality, category) blocks, each sorted by month. Runs are
    labelled on the raw file before filtering to the total category, with assertions that the
    legacy filter-first reconstruction gives the same 2,016 series and that no two series coincide.
    """
    p = sbx_data.load_panel()
    panel = pd.DataFrame(p.values, index=pd.Index(p.meta["run_id"], name="series_id"), columns=p.periods)
    meta = p.meta.set_index("run_id")
    meta.index.name = "series_id"
    return panel, meta


def local_sng2(wide: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Local seasonal-growth component.

    yhat(T+h) = y(T+h-12) * exp(mean of last two observed YoY log growth rates).
    """
    lg = np.log(wide)
    yoy = lg - lg.shift(12, axis=1)
    g = yoy.iloc[:, -2:].mean(axis=1)
    return pd.DataFrame(
        {h: wide.iloc[:, -12 + h - 1] * np.exp(g) for h in horizons}, index=wide.index
    )


def panel_factor6(wide: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Panel common-factor component with six-month YoY momentum."""
    logw = np.log(wide)
    factor = logw.median(axis=0)
    dev_last = logw.iloc[:, -1] - factor.iloc[-1]
    factor_yoy = factor - factor.shift(12)
    g = factor_yoy.iloc[-6:].mean()
    out = {}
    for h in horizons:
        factor_target = factor.iloc[-12 + h - 1] + g
        out[h] = np.exp(dev_last + factor_target)
    return pd.DataFrame(out, index=wide.index)


def exact_backtest() -> tuple[pd.DataFrame, dict]:
    panel, meta = load_panel()
    rows = []
    for origin in ORIGINS:
        train = panel.loc[:, :origin]
        local = local_sng2(train, HORIZONS)
        factor = panel_factor6(train, HORIZONS)
        ensemble = 0.5 * local + 0.5 * factor
        for h in HORIZONS:
            target = origin + pd.DateOffset(months=h)
            y = panel[target]
            block = pd.DataFrame(
                {
                    "series_id": panel.index,
                    "mo": meta["mo"].reindex(panel.index).to_numpy(),
                    "origin": origin,
                    "h": h,
                    "target_date": target,
                    "y": y.to_numpy(),
                    "local_sng2": local[h].to_numpy(),
                    "panel_factor6": factor[h].to_numpy(),
                    "yhat": ensemble[h].to_numpy(),
                }
            )
            rows.append(block)
    pred = pd.concat(rows, ignore_index=True)
    pred["ae"] = (pred["y"] - pred["yhat"]).abs()
    pred["y_prev_year"] = [
        panel.loc[sid, td - pd.DateOffset(months=12)]
        for sid, td in zip(pred["series_id"], pred["target_date"])
    ]
    pred["growth"] = np.log(pred["y"] / pred["y_prev_year"])
    pred["growth_hat"] = np.log(pred["yhat"] / pred["y_prev_year"])

    def r2(a: pd.Series, b: pd.Series) -> float:
        return float(r2_score(a, b))

    per_h = []
    for h, g in pred.groupby("h"):
        per_h.append(
            {
                "h": int(h),
                "n": int(len(g)),
                "MAE": float(g["ae"].mean()),
                "R2_level": r2(g["y"], g["yhat"]),
                "R2_growth": r2(g["growth"], g["growth_hat"]),
                "wMAPE_pct": float(100 * g["ae"].sum() / g["y"].abs().sum()),
            }
        )

    summary = {
        "n_series": int(panel.shape[0]),
        "origins": [str(x.date()) for x in ORIGINS],
        "horizons": HORIZONS,
        "n_pairs": int(len(pred)),
        "model": "0.5 * local_sng2 + 0.5 * panel_factor6",
        "MAE": float(pred["ae"].mean()),
        "R2_level": r2(pred["y"], pred["yhat"]),
        "R2_growth": r2(pred["growth"], pred["growth_hat"]),
        "wMAPE_pct": float(100 * pred["ae"].sum() / pred["y"].abs().sum()),
        "per_h": per_h,
        "public_prophet_mae": 1428,
        "public_reference_ensemble_mae": 762,
    }
    return pred, summary


if __name__ == "__main__":
    predictions, summary = exact_backtest()
    predictions.to_csv(OUT / "exact_backtest_predictions.csv", index=False)
    (OUT / "exact_benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
