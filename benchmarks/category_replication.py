"""Out-of-sample replication of the FROZEN final models on the five category panels.

Every design decision (V3 error feedback, V3-hedge, interval calibration window) was made while
looking at the total-category panel, including the "other" origins. The five category panels
(Здоровье, Маркетплейсы, Общественное питание, Продовольствие, Транспорт) were never used for any
choice, so they are the cleanest available check of whether the gains generalise.

Models are applied unchanged (national context = total national spending, as frozen).
Output: outputs/category_replication.csv, outputs/category_replication_tests.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx import backtest as B  # noqa: E402
from sbx import models as M  # noqa: E402
from sbx.data import ROOT, TOTAL, load_panel, national_monthly, weekly_monthly  # noqa: E402
from sbx.final import FINAL_BENCHMARK, FINAL_ROBUST, final_models, reference_models  # noqa: E402

OUT = ROOT / "outputs"
CATEGORIES = [TOTAL, "Здоровье", "Маркетплейсы", "Общественное питание", "Продовольствие", "Транспорт"]
WINDOWS = {
    "exact": B.EXACT_ORIGINS,
    "other": pd.to_datetime(["2024-04-01", "2024-05-01", "2024-10-01", "2024-11-01"]),
    "apr_nov": pd.date_range("2024-04-01", "2024-11-01", freq="MS"),
    "feb_mar": pd.to_datetime(["2024-02-01", "2024-03-01"]),
}


def models() -> dict:
    m = {**reference_models(), **final_models()}
    m["common_weekly2"] = M.blend_components(M.g_window(2), M.g_weekly(2))
    return m


def main():
    nat, wk = national_monthly(), weekly_monthly()
    ctx = lambda o: {"origin": o, "national": nat[nat.index <= o], "weekly": wk[wk.index <= o]}  # noqa: E731
    rows, tests = [], []
    for cat in CATEGORIES:
        panel = load_panel(cat)
        ms = models()
        df = B.run(panel, ms, pd.date_range("2024-02-01", "2024-11-01", freq="MS"), ctx_fn=ctx)
        for w, os_ in WINDOWS.items():
            sub = df[df["origin"].isin(os_)]
            for name in ms:
                rows.append({"category": cat, "window": w, "model": name, **B.metrics(sub, name)})
            for name in [FINAL_BENCHMARK, FINAL_ROBUST, "v3_hedge_no_ef", "common_national2", "common_weekly2"]:
                t = B.block_bootstrap_diff(sub, name, "v2_ensemble")
                tests.append({"category": cat, "window": w, "model": name, "vs": "v2_ensemble",
                              "dMAE": t["diff"], "dMAE_pct": 100 * t["diff"] / B.metrics(sub, "v2_ensemble")["MAE"],
                              "months_better": t["months_better"], "months_total": t["months_total"],
                              "sign_test_p": t["sign_test_p_one_sided"],
                              "ci95_series_lo": t["ci95_by_series"][0], "ci95_series_hi": t["ci95_by_series"][1]})
        print("done", cat, flush=True)
    res, tst = pd.DataFrame(rows), pd.DataFrame(tests)
    res.to_csv(OUT / "category_replication.csv", index=False)
    tst.to_csv(OUT / "category_replication_tests.csv", index=False)
    pd.set_option("display.width", 220)
    piv = res[res["window"].isin(["exact", "other", "feb_mar"])].pivot_table(index=["category", "model"], columns="window", values="MAE")
    print(piv.round(1).to_string())
    print(tst.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
