#!/usr/bin/env bash
# One-command reproduction of the final submission (~10 min on 4 cores, excluding optional refits).
#
# All in-house results are recomputed from data/raw/. Prophet and TimesFM may use frozen cached
# predictions by default because full refits are expensive; their benchmark scripts document refit.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

$PY benchmarks/exact_backtest.py > /dev/null
$PY tests/test_time_safety.py
$PY tests/test_news_time_safety.py

if [[ "${REFIT_PROPHET:-0}" == "1" ]]; then
  $PY benchmarks/prophet_baseline.py
  $PY benchmarks/prophet_baseline.py --variant log
  $PY benchmarks/prophet_horizons.py
fi

$PY src/pipeline.py > outputs/pipeline.log
$PY benchmarks/rolling_eval.py > outputs/rolling_eval.log
$PY benchmarks/category_replication.py > outputs/category_replication.log
$PY benchmarks/horizons_benchmark.py > outputs/horizons_benchmark.log
$PY benchmarks/early_warning_eval.py > outputs/early_warning_eval.log
$PY benchmarks/early_warning_v2.py > outputs/early_warning_v2.log
$PY benchmarks/news_ablation_global.py > outputs/news_ablation.log
$PY benchmarks/forecast_final.py > outputs/forecast_final.log
$PY benchmarks/real_examples.py > outputs/real_examples.log
$PY benchmarks/build_final.py > outputs/build_final.log
$PY benchmarks/finalize_submission.py > outputs/finalize_submission.log

$PY tests/check_presentation_numbers.py
$PY tests/check_final_additions.py

echo "done: forecasting + horizons + TimesFM cache + early warning + news ablation + real examples + figures"
