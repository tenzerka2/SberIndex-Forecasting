#!/usr/bin/env bash
# One-command reproduction of every number, table and figure in FINAL_REPORT.md (~7 min, 4 cores).
# Prerequisite: the five SberIndex CSVs unpacked into data/raw/ (see data/README.md).
# Prophet is not re-fitted by default (~7 s per fit); the cached sample in
# outputs/prophet_predictions.csv.gz is used. Set REFIT_PROPHET=1 to refit it.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

$PY benchmarks/exact_backtest.py > /dev/null          # V2 contest benchmark, MAE 725.686835
$PY tests/test_time_safety.py                         # leakage / reconstruction tests
if [[ "${REFIT_PROPHET:-0}" == "1" ]]; then
  $PY benchmarks/prophet_baseline.py
  $PY benchmarks/prophet_baseline.py --variant log
fi
$PY benchmarks/rolling_eval.py > outputs/rolling_eval.log         # all models, 10 origins
$PY benchmarks/category_replication.py > outputs/category_replication.log
$PY benchmarks/early_warning_eval.py > outputs/early_warning_eval.log
$PY benchmarks/early_warning_v2.py > outputs/early_warning_v2.log
$PY benchmarks/forecast_final.py > outputs/forecast_final.log     # 2025 forecasts + intervals
$PY benchmarks/build_final.py > outputs/build_final.log           # FINAL_METRICS.csv, ABLATION.csv, figures/
echo "done: FINAL_METRICS.csv ABLATION.csv figures/ outputs/final_forecasts_2025.csv.gz"
