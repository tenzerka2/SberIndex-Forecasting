"""Benchmark on the contest horizons h = 1, 3, 6, 12 (models frozen, identical pairs per horizon).

Evidence base is set by the data (24 months, 2023-01..2024-12) and the target must be observed:
  h = 1, 3, 6  origins 2024-02 .. 2024-(12-h): V3 / V3-hedge need 13 months of history (2-month YoY);
               10 / 8 / 5 origins.
  h = 12       target <= 2024-12 forces origin <= 2023-12, where no YoY growth is observable yet:
               V3, V3-hedge and seasonal growth are NOT DEFINED. Prophet was not evaluated
               here: short history limits seasonal estimation but does not prohibit a fit. Evaluated on origins
               2023-01..2023-12 with the models that are defined there: seasonal naive, seasonal naive
               x national YoY (national SberIndex series, data <= origin) and zero-shot TimesFM.
Prophet (log target, cached 400-series sample) and TimesFM (cached zero-shot predictions) are merged on
identical pairs; Prophet rows are reported on the 400-series sample next to all other models.
Outputs: outputs/horizons_metrics.csv, outputs/horizons_intervals.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx import backtest as B  # noqa: E402
from sbx import models as M  # noqa: E402
from sbx.data import ROOT, load_panel, national_monthly, weekly_monthly  # noqa: E402
from sbx.final import final_models  # noqa: E402
from sbx.intervals import conformal  # noqa: E402

OUT = ROOT / "outputs"
H_MAIN = [1, 3, 6]
LIMITS = {
    1: "10 origins (2024-02..2024-11), 10 target months",
    3: "8 origins (2024-02..2024-09), 8 target months",
    6: "5 origins (2024-02..2024-06), 5 target months (2024-08..2024-12); origins inside the Q1 anomaly",
    12: "12 origins (2023-01..2023-12), targets 2024-01..2024-12 incl. the Q1 anomaly; V3/V3-hedge/"
        "seasonal growth undefined (no observable YoY at origin); Prophet not evaluated",
}


def metrics_rows(df, cols, h, subset):
    rows = []
    for c in cols:
        d = df.dropna(subset=[c])
        m = B.metrics(d, c)
        rows.append({"h": h, "subset": subset, "model": c, "n_origins": int(d["origin"].nunique()),
                     "n_pairs": int(len(d)), **m, "evidence": LIMITS[h]})
    return rows


def main():
    p = load_panel()
    nat, wk = national_monthly(), weekly_monthly()
    ctx = lambda o: {"origin": o, "national": nat[nat.index <= o], "weekly": wk[wk.index <= o]}  # noqa: E731
    models = {"seasonal_naive": M.seasonal_naive, "seasonal_growth1": M.seasonal_growth(1),
              "v2_ensemble": M.v2_ensemble, **final_models()}
    df = B.run(p, models, pd.date_range("2024-02-01", "2024-11-01", freq="MS"), horizons=H_MAIN, ctx_fn=ctx)

    tfm = pd.read_csv(OUT / "timesfm_predictions.csv.gz", parse_dates=["origin"])
    df = df.merge(tfm, on=["series", "origin", "h"], how="left")
    pf = OUT / "prophet_horizons.csv.gz"
    has_prophet = pf.exists()
    if has_prophet:
        df = df.merge(pd.read_csv(pf, parse_dates=["origin"]), on=["series", "origin", "h"], how="left")
        df["prophet_log"] = df["prophet_log"].clip(lower=1.0)

    # h = 12 on the only origins where its target is observable
    rows12 = []
    L, Y = p.logs, p.values
    s = np.log(nat["nat_spend"])
    nat_yoy = (s - s.shift(12))
    for T in range(0, 12):
        o = p.periods[T]
        rows12.append(pd.DataFrame({"series": np.arange(Y.shape[0]), "origin": o, "h": 12,
                                    "target_date": p.periods[T + 12], "y": Y[:, T + 12], "y_base": Y[:, T],
                                    "seasonal_naive": Y[:, T],
                                    "naive_x_national_yoy": Y[:, T] * np.exp(float(nat_yoy.loc[o]))}))
    d12 = pd.concat(rows12, ignore_index=True).merge(tfm, on=["series", "origin", "h"], how="left")

    base_cols = ["seasonal_naive", "seasonal_growth1", "v2_ensemble", "v3", "v3_hedge", "timesfm_raw", "timesfm_yoy"]
    rows = []
    for h in H_MAIN:
        g = df[df["h"] == h]
        rows += metrics_rows(g, base_cols, h, "all_2016_series")
        if has_prophet:
            gs = g.dropna(subset=["prophet_log"])
            rows += metrics_rows(gs, base_cols + ["prophet_log"], h, "prophet_sample_400")
    rows += metrics_rows(d12, ["seasonal_naive", "naive_x_national_yoy", "timesfm_raw"], 12, "all_2016_series")
    for m in ["seasonal_growth1", "v2_ensemble", "v3", "v3_hedge", "timesfm_yoy", "prophet_log"]:
        rows.append({"h": 12, "subset": "all_2016_series", "model": m, "n_origins": 0, "n_pairs": 0,
                     "evidence": "not defined: needs observable YoY growth at origin (origin >= 2024-01/02), "
                                 "but a 12-month target must be <= 2024-12"})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "horizons_metrics.csv", index=False, float_format="%.4f")

    # 80% interval coverage: V3 two-part conformal vs TimesFM quantiles (h = 1, 3, 6)
    iv = []
    for col in ["v3", "v3_hedge"]:
        c = conformal(df[["series", "origin", "h", "target_date", "y", col]], col, L, p.periods, levels=(0.8,))
        for h in H_MAIN:
            g = c[(c["h"] == h)].dropna(subset=[f"{col}_lo80"])
            iv.append({"model": col, "h": h, "n_pairs": len(g),
                       "coverage80": float(((g.y >= g[f"{col}_lo80"]) & (g.y <= g[f"{col}_hi80"])).mean()),
                       "mean_width80_rub": float((g[f"{col}_hi80"] - g[f"{col}_lo80"]).mean()),
                       "note": "pairs with >= 1000 calibration pairs of target <= origin"})
    for v in ["timesfm_raw", "timesfm_yoy"]:
        for h in H_MAIN:
            g = df[(df["h"] == h)].dropna(subset=[v])
            iv.append({"model": v, "h": h, "n_pairs": len(g),
                       "coverage80": float(((g.y >= g[f"{v}_lo80"]) & (g.y <= g[f"{v}_hi80"])).mean()),
                       "mean_width80_rub": float((g[f"{v}_hi80"] - g[f"{v}_lo80"]).mean()),
                       "note": "model quantiles q10..q90, zero-shot"})
    g = d12.dropna(subset=["timesfm_raw"])
    iv.append({"model": "timesfm_raw", "h": 12, "n_pairs": len(g),
               "coverage80": float(((g.y >= g["timesfm_raw_lo80"]) & (g.y <= g["timesfm_raw_hi80"])).mean()),
               "mean_width80_rub": float((g["timesfm_raw_hi80"] - g["timesfm_raw_lo80"]).mean()),
               "note": "model quantiles q10..q90, zero-shot"})
    ivd = pd.DataFrame(iv)
    ivd.to_csv(OUT / "horizons_intervals.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print(res[res["subset"] == "all_2016_series"][["h", "model", "n_origins", "n_pairs", "MAE", "R2_level", "R2_growth", "wMAPE_pct"]].round(3).to_string(index=False))
    if has_prophet:
        print(res[res["subset"] == "prophet_sample_400"][["h", "model", "n_pairs", "MAE", "R2_level", "R2_growth"]].round(3).to_string(index=False))
    print(ivd.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
