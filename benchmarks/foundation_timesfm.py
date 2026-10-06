"""Zero-shot TimesFM 2.5-200M (Google, frozen weights) on the same forecast pairs as our models.

No fine-tuning, no hyper-parameter search; two variants fixed in advance:
  timesfm_raw  context = the series' levels up to the origin (1..23 points)
  timesfm_yoy  context = the series' log YoY growth up to the origin (needs origin >= 2024-02,
               like V3); forecast growth is applied to the level of the same month a year earlier
Quantile head: q10 / q90 -> 80% interval (the model has no 5% / 95% quantiles).

Origins 2023-01..2024-11, all horizons h <= 12 with target <= 2024-12.
Weights: python benchmarks/fetch_timesfm.py (official Google public bucket; Hugging Face is
blocked in our environment). Runs in its own venv because of JAX:
    python -m venv .venv-tsfm && .venv-tsfm/bin/pip install -r requirements-tsfm.txt
    .venv-tsfm/bin/python benchmarks/foundation_timesfm.py
Output: outputs/timesfm_predictions.csv.gz
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.data import ROOT, load_panel  # noqa: E402

CKPT = ROOT / "models" / "timesfm-2.5-200m-flax"
OUT = ROOT / "outputs" / "timesfm_predictions.csv.gz"
MAX_H = 12


def load_model():
    import timesfm

    m = timesfm.TimesFM_2p5_200M_flax.from_pretrained(str(CKPT.resolve()))
    m.compile(timesfm.ForecastConfig(max_context=32, max_horizon=MAX_H, normalize_inputs=True, per_core_batch_size=256,
                                     use_continuous_quantile_head=True, force_flip_invariance=True,
                                     infer_is_positive=True, fix_quantile_crossing=True))
    return m


def main():
    t0 = time.time()
    p = load_panel()
    Y, L = p.values, p.logs
    n, last = Y.shape[0], Y.shape[1] - 1
    model = load_model()
    frames = []
    for T in range(0, last):
        hs = [h for h in range(1, MAX_H + 1) if T + h <= last]
        pt, q = model.forecast(horizon=MAX_H, inputs=[Y[i, : T + 1].astype(float) for i in range(n)])
        block = {"raw": (pt, q[:, :, 1], q[:, :, 9])}
        if T >= 13:
            yoy = L[:, 12 : T + 1] - L[:, : T - 11]
            pty, qy = model.forecast(horizon=MAX_H, inputs=[yoy[i].astype(float) for i in range(n)])
            base = np.stack([Y[:, T + h - 12] for h in range(1, MAX_H + 1)], axis=1)
            block["yoy"] = (base * np.exp(pty), base * np.exp(qy[:, :, 1]), base * np.exp(qy[:, :, 9]))
        for h in hs:
            row = {"series": np.arange(n), "origin": p.periods[T], "h": h}
            for v, (a, lo, hi) in block.items():
                row[f"timesfm_{v}"] = a[:, h - 1]
                row[f"timesfm_{v}_lo80"] = lo[:, h - 1]
                row[f"timesfm_{v}_hi80"] = hi[:, h - 1]
            frames.append(pd.DataFrame(row))
        print("origin", p.periods[T].date(), round(time.time() - t0, 1), "s", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT, index=False, float_format="%.2f", compression={"method": "gzip", "mtime": 0})
    print("rows", len(out), "->", OUT)


if __name__ == "__main__":
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
    main()
