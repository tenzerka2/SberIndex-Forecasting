#!/usr/bin/env bash
# Reproduce the 2026-10-07 audit with NumPy and pandas; legacy outputs are untouched.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
mkdir -p outputs/research
$PY tests/test_research.py
$PY benchmarks/research_forecasts.py --categories
$PY benchmarks/research_national_shape.py
$PY benchmarks/research_v4.py
$PY benchmarks/research_warning.py
$PY benchmarks/research_news.py
$PY benchmarks/research_summary.py
$PY benchmarks/research_forecast_export.py --model v4_diversified
echo "Research metrics are in outputs/research/. See IMPROVEMENT_REPORT.md for limits."
