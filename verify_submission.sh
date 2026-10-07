#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export OPENBLAS_NUM_THREADS=1
"$PY" benchmarks/verify_submission.py
"$PY" benchmarks/submission_warning_cases.py
"$PY" -m unittest discover -s tests -p test_operational_horizons.py
"$PY" -m unittest discover -s tests -p test_contest_timesfm.py
