#!/usr/bin/env bash
# One-command reproduction of FINAL_REPORT.md / PRESENTATION_DATA.md (~8 min on 4 cores).
#
# Everything produced by our models is recomputed from the raw CSVs in data/raw/:
# reconstruction, all backtests, category replication, early warning, intervals, 2025 forecasts,
# FINAL_METRICS.csv, ABLATION.csv, figures/ and outputs/presentation_sources.json.
#
# The only exception is the Prophet baseline (400-series sample, ~7 s per cmdstan fit, ~1 h in
# total). By default its cached predictions in outputs/prophet_predictions.csv.gz are re-scored
# against freshly computed targets; run   REFIT_PROPHET=1 ./reproduce.sh   to refit it from scratch.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

$PY benchmarks/exact_backtest.py > /dev/null                        # V2 contest benchmark, MAE 725.686835
$PY tests/test_time_safety.py                                       # 7 leakage / reconstruction tests
if [[ "${REFIT_PROPHET:-0}" == "1" ]]; then
  $PY benchmarks/prophet_baseline.py                                # default Prophet, sample of 400 series
  $PY benchmarks/prophet_baseline.py --variant log                  # log-target Prophet, same sample
fi
$PY src/pipeline.py > outputs/pipeline.log                          # legacy pooled LightGBM + honest legacy shock metrics
$PY benchmarks/rolling_eval.py > outputs/rolling_eval.log           # all models, 10 origins, bootstrap, intervals
$PY benchmarks/category_replication.py > outputs/category_replication.log
$PY benchmarks/early_warning_eval.py > outputs/early_warning_eval.log   # fixed-split EW (first iteration)
$PY benchmarks/early_warning_v2.py > outputs/early_warning_v2.log   # rolling EW (final)
$PY benchmarks/forecast_final.py > outputs/forecast_final.log       # 2025 forecasts + intervals
$PY benchmarks/build_final.py > outputs/build_final.log             # FINAL_METRICS, ABLATION, figures, sources
$PY tests/check_presentation_numbers.py                             # quoted numbers == outputs
echo "done: FINAL_METRICS.csv ABLATION.csv figures/ outputs/presentation_sources.json outputs/final_forecasts_2025.csv.gz"
