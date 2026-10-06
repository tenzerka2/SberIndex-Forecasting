"""Full time-safe evaluation: baselines, V2, new variants, ablation, bootstrap, intervals.

Windows (all forecasts use data <= origin only):
  exact   origins 2024-06..2024-09, h = 1..3 (the contest protocol, 24,192 pairs)
  other   origins 2024-04, 2024-05, 2024-10, 2024-11 (h truncated at 2024-12): never used for design
  apr_nov all origins 2024-04..2024-11
Origins 2024-02/03 are also computed but reported separately: their windows sit inside the
Q1-2023/Q1-2024 level anomaly of the municipal panel (see REPORT.md).

Usage:  python benchmarks/rolling_eval.py [--fast]   (--fast skips the slow pooled GBMs)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx import backtest as B  # noqa: E402
from sbx import models as M  # noqa: E402
from sbx.data import ROOT, load_panel, national_monthly  # noqa: E402
from sbx.intervals import conformal, interval_metrics  # noqa: E402

OUT = ROOT / "outputs"
WINDOWS = {
    "exact": B.EXACT_ORIGINS,
    "other": pd.to_datetime(["2024-04-01", "2024-05-01", "2024-10-01", "2024-11-01"]),
    "apr_nov": pd.date_range("2024-04-01", "2024-11-01", freq="MS"),
    "feb_mar_anomaly": pd.to_datetime(["2024-02-01", "2024-03-01"]),
}
FINAL = "v3_error_feedback"
REFERENCE = "v2_ensemble"


def build_models(fast: bool) -> dict:
    g = M.g_window
    m = {
        # baselines / references
        "seasonal_naive": M.seasonal_naive,
        "seasonal_growth1": M.seasonal_growth(1),
        "local_sng2": M.local_sng2,
        "factor_only6": M.panel_factor6,
        "v2_ensemble": M.v2_ensemble,
        # common-factor variants (V2 idiosyncratic structure, different common growth)
        "common_g2": M.blend_components(g(2), g(2)),
        "common_adaptive": M.blend_components(M.g_adaptive(), M.g_adaptive()),
        "common_national2": M.blend_components(g(2), M.g_national(2)),
        # weights / decomposition
        "horizon_weights": M.horizon_weights(),
        "lowrank_factor4": M.lowrank_factor(4, 2),
        # final candidate: V2 + online error feedback
        "v3_error_feedback": M.error_feedback(),
        "v3_ef_lag1": M.error_feedback(lags=1),
        "v3_ef_adaptive_common": M.error_feedback(base=M.blend_components(M.g_adaptive(), M.g_adaptive())),
        # robust alternative: hedge of common-growth estimators (pre-specified, no fitted weights)
        "v3_hedge": M.v3_hedge,
        "v3_hedge_ef": M.error_feedback(base=M.v3_hedge),
        "ef_common_national2": M.error_feedback(base=M.blend_components(g(2), M.g_national(2))),
    }
    if not fast:
        m.update({
            "residual_lgbm": M.pooled_gbm("lgbm", residual=True),
            "residual_catboost": M.pooled_gbm("catboost", residual=True),
            "pooled_lgbm_direct": M.pooled_gbm("lgbm", residual=False),
        })
    return m


ABLATION = [
    ("seasonal_naive", "last-year level only"),
    ("seasonal_growth1", "+ own YoY of last month"),
    ("local_sng2", "own YoY averaged over 2 months"),
    ("factor_only6", "panel factor only (6m common momentum)"),
    ("v2_ensemble", "V2: 0.5 local_sng2 + 0.5 factor6"),
    ("horizon_weights", "V2 + horizon weights learned on past origins"),
    ("common_g2", "V2 with 2m common momentum in both parts"),
    ("common_adaptive", "V2 with adaptively chosen common window"),
    ("v3_ef_lag1", "V2 + error feedback (1 lag)"),
    ("v3_error_feedback", "V2 + error feedback (2 lags)  [final]"),
    ("v3_ef_adaptive_common", "V2 + adaptive common + error feedback"),
    ("common_national2", "V2, panel part advanced by national 2m YoY"),
    ("ef_common_national2", "national common + error feedback"),
    ("v3_hedge", "V2, panel part advanced by hedge {muni 6m, muni 2m, national 2m}"),
    ("v3_hedge_ef", "hedge + error feedback  [robust alternative]"),
    ("residual_lgbm", "V2 + LightGBM residual"),
    ("residual_catboost", "V2 + CatBoost residual"),
    ("pooled_lgbm_direct", "pooled LightGBM, direct YoY target"),
    ("lowrank_factor4", "rank-4 SVD factor model"),
]


def main(fast: bool = False):
    t0 = time.time()
    panel = load_panel()
    nat = national_monthly()
    models = build_models(fast)
    origins = pd.date_range("2024-02-01", "2024-11-01", freq="MS")
    df = B.run(panel, models, origins, ctx_fn=lambda o: {"origin": o, "national": nat[nat.index <= o]})
    print("backtest", round(time.time() - t0, 1), "s", flush=True)
    names = list(models)

    # exactness check against the published benchmark
    ex = df[df["origin"].isin(B.EXACT_ORIGINS)]
    v2 = B.metrics(ex, REFERENCE)
    assert len(ex) == 24192 and abs(v2["MAE"] - 725.6868352098129) < 1e-6, v2

    tables = {}
    for w, os_ in WINDOWS.items():
        tables[w] = B.table(df[df["origin"].isin(os_)], names)
    per_h = B.table(ex, names, by="h")
    per_origin = B.table(df, names, by="origin").pivot(index="model", columns="origin", values="MAE")
    per_origin.columns = [c.strftime("%Y-%m") for c in per_origin.columns]

    # paired bootstrap vs V2
    boot = {}
    for w in ["exact", "other", "apr_nov"]:
        sub = df[df["origin"].isin(WINDOWS[w])]
        boot[w] = {c: B.block_bootstrap_diff(sub, c, REFERENCE) for c in names if c != REFERENCE}

    # ablation table
    abl = []
    for key, desc in ABLATION:
        if key not in names:
            continue
        row = {"model": key, "description": desc}
        for w in ["exact", "other", "apr_nov"]:
            t = tables[w].set_index("model").loc[key]
            row[f"MAE_{w}"] = t["MAE"]
        row["R2_growth_exact"] = tables["exact"].set_index("model").loc[key, "R2_growth"]
        row["wMAPE_exact"] = tables["exact"].set_index("model").loc[key, "wMAPE_pct"]
        row["dMAE_vs_V2_exact"] = row["MAE_exact"] - tables["exact"].set_index("model").loc[REFERENCE, "MAE"]
        row["dMAE_vs_V2_other"] = row["MAE_other"] - tables["other"].set_index("model").loc[REFERENCE, "MAE"]
        abl.append(row)
    abl = pd.DataFrame(abl)

    # conformal intervals for V2 and the final model
    iv = {}
    for col in [REFERENCE, FINAL, "v3_hedge_ef"]:
        c = conformal(df[["series", "origin", "h", "target_date", "y", col]], col, panel.logs, panel.periods)
        iv[col] = {w: interval_metrics(c[c["origin"].isin(WINDOWS[w])], col) for w in ["exact", "other", "apr_nov"]}

    # Prophet (cached sample of 400 series) on identical pairs
    prophet = None
    pf = OUT / "prophet_predictions.csv.gz"
    if pf.exists():
        pr = pd.read_csv(pf, parse_dates=["origin"])
        j = ex.merge(pr, on=["series", "origin", "h"])
        pcols = [c for c in ["prophet", "prophet_log"] if c in j]
        j = j.dropna(subset=pcols)
        for c in pcols:
            j[c] = j[c].clip(lower=1.0)
        prophet = {c: B.metrics(j, c) for c in pcols + ["seasonal_naive", "factor_only6", REFERENCE, FINAL, "v3_hedge_ef"]}
        prophet["prophet_median_ae"] = float((j["prophet"] - j["y"]).abs().median())
        prophet["v2_median_ae"] = float((j[REFERENCE] - j["y"]).abs().median())
        prophet["n_series_sampled"] = int(j["series"].nunique())

    df.to_csv(OUT / "rolling_predictions.csv.gz", index=False)
    for w, t in tables.items():
        t.to_csv(OUT / f"metrics_{w}.csv", index=False)
    per_h.to_csv(OUT / "metrics_exact_by_h.csv", index=False)
    per_origin.to_csv(OUT / "mae_by_origin.csv")
    abl.to_csv(OUT / "ablation.csv", index=False)
    summary = {
        "final_model": FINAL,
        "reference": REFERENCE,
        "exact_reproduced": v2,
        "final_exact": B.metrics(ex, FINAL),
        "final_exact_by_h": B.table(ex, [FINAL], by="h").to_dict("records"),
        "windows": {w: tables[w].set_index("model")[["MAE", "R2_growth", "wMAPE_pct"]].round(4).to_dict("index") for w in tables},
        "bootstrap_vs_v2": boot,
        "intervals": iv,
        "prophet_sample": prophet,
    }
    (OUT / "rolling_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(abl.round(2).to_string(index=False))
    print(per_origin.round(0).to_string())
    for w in ["exact", "other", "apr_nov"]:
        b = boot[w][FINAL]
        print(w, FINAL, "vs V2:", {k: (np.round(v, 2) if not isinstance(v, int) else v) for k, v in b.items()})
    for col in iv:
        for w in iv[col]:
            for r in iv[col][w]:
                print("interval", col, w, {k: round(v, 3) for k, v in r.items()})
    if prophet:
        print("prophet sample", {k: (round(v["MAE"], 1) if isinstance(v, dict) else round(v, 1)) for k, v in prophet.items()})
    print("total", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main(fast="--fast" in sys.argv)
