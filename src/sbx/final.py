"""Frozen final forecasting models. Do not tune these on the exact window.

FINAL_BENCHMARK (V3)   = V2 ensemble + error feedback (2 lags)
FINAL_ROBUST (V3-hedge) = local_sng2 + panel part advanced by the mean of
                          {municipal 6m, municipal 2m, national 2m} common YoY, + error feedback

Frozen on 2026-10-06 (commit 80d1384). Any later change must be re-validated on the untouched
category panels (benchmarks/category_replication.py) and documented in FINAL_REPORT.md.
"""
from __future__ import annotations

from . import models as M

FINAL_BENCHMARK = "v3"
FINAL_ROBUST = "v3_hedge"


def final_models() -> dict:
    return {
        FINAL_BENCHMARK: M.error_feedback(base=M.v2_ensemble, lags=2),
        FINAL_ROBUST: M.error_feedback(base=M.v3_hedge, lags=2),
    }


def reference_models() -> dict:
    """Components and baselines reported next to the finals."""
    return {
        "seasonal_naive": M.seasonal_naive,
        "seasonal_growth1": M.seasonal_growth(1),
        "local_sng2": M.local_sng2,
        "factor_only6": M.panel_factor6,
        "v2_ensemble": M.v2_ensemble,
        "v3_hedge_no_ef": M.v3_hedge,
        "common_national2": M.blend_components(M.g_window(2), M.g_national(2)),
    }
