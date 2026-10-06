"""Log-target Prophet on the horizon benchmark origins (2024-02..2024-11), h = 1..6, same fixed
400-series sample as prophet_baseline.py (seed 0). One fit per (series, origin) gives all horizons.

h = 12 is not run: the only origins with an observable 12-month-ahead target have <= 12 months of
history, where a yearly seasonality cannot be identified.
Output: outputs/prophet_horizons.csv.gz (cached; refit with REFIT_PROPHET=1 ./reproduce.sh).
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prophet_baseline import _fit_one  # noqa: E402
from sbx.data import ROOT, load_panel, month_index  # noqa: E402

OUT = ROOT / "outputs" / "prophet_horizons.csv.gz"
ORIGINS = pd.date_range("2024-02-01", "2024-11-01", freq="MS")


def main(workers: int = max(1, (os.cpu_count() or 2) - 1), sample: int = 400):
    panel = load_panel()
    ids = np.sort(np.random.default_rng(0).choice(panel.values.shape[0], sample, replace=False))
    rows = []
    for origin in ORIGINS:
        T = month_index(origin)
        hs = [h for h in range(1, 7) if T + h <= 23]
        ds = list(panel.periods[: T + 1])
        tasks = [(int(i), ds, panel.values[i, : T + 1], hs, "log") for i in ids]
        with ProcessPoolExecutor(workers) as ex:
            for i, yhat in ex.map(_fit_one, tasks, chunksize=16):
                rows.extend((i, origin, h, v) for h, v in zip(hs, yhat))
        print("done", origin.date(), flush=True)
    pd.DataFrame(rows, columns=["series", "origin", "h", "prophet_log"]).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
