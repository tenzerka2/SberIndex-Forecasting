"""Check that the key numbers quoted in PRESENTATION.md / PRESENTATION_DATA.md / JURY_QA.md match the
reproducible outputs (rounded as quoted). Run after ./reproduce.sh; exits non-zero on any mismatch."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "outputs"
fails: list[str] = []


def check(name, value, expected, nd):
    if round(float(value), nd) != round(float(expected), nd):
        fails.append(f"{name}: output {value} != quoted {expected}")


fm = pd.read_csv(ROOT / "FINAL_METRICS.csv")
g = lambda w, m, c="MAE", h="1-3": fm[(fm.window == w) & (fm.model == m) & (fm.h.astype(str) == h)][c].iloc[0]  # noqa: E731
check("V3 MAE exact", g("exact", "v3"), 719.87, 2)
check("V3 R2 growth", g("exact", "v3", "R2_growth"), 0.438, 3)
check("V3 wMAPE", g("exact", "v3", "wMAPE_pct"), 2.28, 2)
check("V2 MAE exact", g("exact", "v2_ensemble"), 725.69, 2)
check("hedge MAE exact", g("exact", "v3_hedge"), 730.14, 2)
check("seasonal naive", g("exact", "seasonal_naive"), 4205.64, 2)
check("seasonal growth", g("exact", "seasonal_growth1"), 976.89, 2)
check("V3 other", g("other", "v3"), 921.67, 2)
check("hedge other", g("other", "v3_hedge"), 792.98, 2)
check("V2 other", g("other", "v2_ensemble"), 925.12, 2)
check("prophet log", g("exact_sample400", "prophet_log"), 1627.15, 2)
check("V3 sample", g("exact_sample400", "v3"), 694.06, 2)
check("long h4 hedge", g("long_h", "v3_hedge", h="4"), 1158.55, 2)

s = json.loads((O / "rolling_summary.json").read_text())
b = s["bootstrap_vs_v2"]["exact"]["v3"]
check("V3-V2 diff", b["diff"], -5.82, 2)
check("months better", b["months_better"], 6, 0)
check("sign test", b["sign_test_p_one_sided"], 0.016, 3)

r = pd.read_csv(O / "category_replication_tests.csv")
c = r[(r.category != "Все категории") & (r.model == "v3") & (r.window == "exact")]
check("V3 categories better", (c.dMAE < 0).sum(), 5, 0)
check("V3 cat-months", c.months_better.sum(), 28, 0)
w = r[(r.category != "Все категории") & (r.model == "common_weekly2") & (r.window == "apr_nov")]
check("weekly worse cats", (w.dMAE > 0).sum(), 5, 0)

ew = pd.read_csv(O / "ew_v2_metrics.csv").set_index(["task", "method"])
check("detect PR-AUC", ew.loc[("detect", "sup_base"), "PR_AUC"], 0.383, 3)
check("detect lift", ew.loc[("detect", "sup_base"), "lift_vs_base_rate"], 21.1, 1)
check("detected share", 100 * ew.loc[("detect", "sup_base"), "detected_share"], 82, 0)
check("FA per 100", ew.loc[("detect", "sup_base"), "false_alarms_per_100"], 2.11, 2)
check("predict best lift", ew.loc[("predict", "bocpd"), "lift_vs_base_rate"], 2.53, 2)

ps = json.loads((O / "presentation_sources.json").read_text())
check("grid better exact", ps["v2_grid"]["configs_better_than_V2_exact"], 8, 0)
check("grid better other", ps["v2_grid"]["configs_better_than_V2_other"], 84, 0)
check("forecast gap", ps["forecast_2025_median_abs_gap_v3_vs_hedge_pct_by_h"]["1"], 0.38, 2)

cf = pd.read_csv(O / "common_factor_vs_national.csv").set_index("month")
check("muni yoy Jan24", cf.loc["2024-01", "muni_median_yoy_pct"], 5.1, 1)
check("nat yoy Jan24", cf.loc["2024-01", "national_yoy_nominal_pct"], 16.3, 1)
lr = pd.read_csv(O / "shift_label_rates.csv").set_index("month")
check("legacy Nov23", lr.loc["2023-11", "legacy_raw_level_pct"], 23.4, 1)
rc = pd.read_csv(O / "ew_v2_rate_correlations.csv")
check("n rate tests", len(rc), 42, 0)
check("min p", rc.p.min(), 0.031, 3)
check("legacy honest PR-AUC", json.loads((O / "shock_metrics.json").read_text())["PR_AUC"], 0.024, 3)
a = json.loads((O / "data_audit.json").read_text())
check("homonyms", a["homonym_series"], 107, 0)
check("linkable", a["category_linkable_series"], 1909, 0)

if fails:
    print("MISMATCH:\n" + "\n".join(fails))
    sys.exit(1)
print("all presentation numbers match outputs")
