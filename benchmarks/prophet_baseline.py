"""Per-series Prophet baseline on the exact contest origins (cached).

Each fit sees only months <= origin. A cmdstan fit costs ~7 s in our sandbox, so by default a fixed
random sample of 400 series (seed 0) is used and compared with other models on the SAME pairs.
Run with --all for the full 2,016 x 4 fits. Output: outputs/prophet_predictions.csv.gz

Variants: "default" (Prophet defaults, yearly seasonality on levels; reproduces the instability of
Prophet with < 2 years of history) and "log" (--variant log: log target, yearly Fourier order 4,
conservative changepoints) -> column prophet_log.
"""
from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.backtest import EXACT_ORIGINS, HORIZONS  # noqa: E402
from sbx.data import ROOT, load_panel, month_index  # noqa: E402

OUT = ROOT / "outputs" / "prophet_predictions.csv.gz"


def _fit_one(args):
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    series, ds, y, hs, variant = args
    if variant == "log":
        m = Prophet(yearly_seasonality=4, weekly_seasonality=False, daily_seasonality=False,
                    changepoint_prior_scale=0.01, uncertainty_samples=0)
        m.fit(pd.DataFrame({"ds": ds, "y": np.log(y)}))
    else:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(pd.DataFrame({"ds": ds, "y": y}))
    fut = pd.DataFrame({"ds": [ds[-1] + pd.DateOffset(months=h) for h in hs]})
    yhat = m.predict(fut)["yhat"].to_numpy()
    return series, np.exp(yhat) if variant == "log" else yhat


def main(workers: int = os.cpu_count() or 2, sample: int | None = 400, variant: str = "default"):
    panel = load_panel()
    n = panel.values.shape[0]
    ids = np.arange(n) if sample is None else np.sort(np.random.default_rng(0).choice(n, sample, replace=False))
    rows = []
    for origin in EXACT_ORIGINS:
        T = month_index(origin)
        hs = [h for h in HORIZONS if T + h <= 23]
        ds = list(panel.periods[: T + 1])
        tasks = [(int(i), ds, panel.values[i, : T + 1], hs, variant) for i in ids]
        with ProcessPoolExecutor(workers) as ex:
            for i, yhat in ex.map(_fit_one, tasks, chunksize=32):
                for h, v in zip(hs, yhat):
                    rows.append((i, origin, h, v))
        print("done", origin.date(), flush=True)
    col = "prophet" if variant == "default" else f"prophet_{variant}"
    new = pd.DataFrame(rows, columns=["series", "origin", "h", col])
    new["origin"] = pd.to_datetime(new["origin"])
    if OUT.exists():
        old = pd.read_csv(OUT, parse_dates=["origin"]).drop(columns=[col], errors="ignore")
        new = old.merge(new, on=["series", "origin", "h"], how="outer")
    new.to_csv(OUT, index=False)


if __name__ == "__main__":
    variant = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else "default"
    main(sample=None if "--all" in sys.argv else 400, variant=variant)
