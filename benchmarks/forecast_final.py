"""Out-of-sample forecasts for 2025 from origin 2024-12 with the two frozen final models.

  * point forecasts for h = 1..12 (2025-01..2025-12);
  * 80% / 90% two-part conformal intervals calibrated on the backtest of the same model
    (origins 2024-02..2024-11, h <= 3, targets <= 2024-12 only);
  * h = 1..3 are backtested; h = 4..6 have limited evidence (see FINAL_METRICS.csv, window
    "long_h"); h = 7..12 are NOT backtestable with 24 months of history and are provided for
    completeness only. Intervals for h > 3 assume sqrt(h) error growth.

Output: outputs/final_forecasts_2025.csv.gz  (series = run_id of the reconstructed series; homonym
municipalities share a name, so run_id is the only unambiguous key).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx import backtest as B  # noqa: E402
from sbx.data import ROOT, load_panel, national_monthly, weekly_monthly  # noqa: E402
from sbx.final import final_models  # noqa: E402
from sbx.intervals import future_intervals  # noqa: E402

ORIGIN = pd.Timestamp("2024-12-01")
H = list(range(1, 13))


def main():
    panel = load_panel()
    nat, wk = national_monthly(), weekly_monthly()
    ctx_fn = lambda o: {"origin": o, "national": nat[nat.index <= o], "weekly": wk[wk.index <= o]}  # noqa: E731
    models = final_models()
    calib = B.run(panel, models, pd.date_range("2024-02-01", "2024-11-01", freq="MS"), ctx_fn=ctx_fn)
    hist = panel.logs.copy()
    hist.setflags(write=False)
    frames = []
    for name, m in models.items():
        pred = m(hist, H, ctx_fn(ORIGIN))
        fut = pd.concat([pd.DataFrame({"series": np.arange(hist.shape[0]), "origin": ORIGIN, "h": h,
                                       "target_date": ORIGIN + pd.DateOffset(months=h), name: np.exp(pred[h])})
                         for h in H], ignore_index=True)
        fut = future_intervals(calib[["series", "origin", "h", "target_date", "y", name]], name, fut,
                               panel.logs, panel.periods)
        frames.append(fut.set_index(["series", "origin", "h", "target_date"]))
    out = pd.concat(frames, axis=1).reset_index()
    out.insert(1, "run_id", panel.meta["run_id"].to_numpy()[out["series"]])
    out.insert(2, "mo", panel.meta["mo"].to_numpy()[out["series"]])
    out.insert(3, "homonym", panel.meta["homonym"].to_numpy()[out["series"]])
    out["evidence"] = np.select([out["h"] <= 3, out["h"] <= 6], ["backtested", "limited"], "not_backtestable")
    out.to_csv(ROOT / "outputs" / "final_forecasts_2025.csv.gz", index=False, float_format="%.1f",
               compression={"method": "gzip", "mtime": 0})  # byte-stable across reruns
    print(out.groupby("h")[["v3", "v3_hedge"]].median().round(0).to_string())
    print(out.head(3).T.to_string())


if __name__ == "__main__":
    main()
