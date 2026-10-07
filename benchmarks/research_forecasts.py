"""Reproduce V3, then evaluate the fixed research candidates without overwriting finals."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.data import ROOT, load_panel, national_monthly
from sbx import backtest as B, models as M
from sbx.final import final_models
from sbx.candidates import research_models

OUT = ROOT / "outputs" / "research"
CATEGORIES = ["Все категории", "Здоровье", "Маркетплейсы", "Общественное питание", "Продовольствие", "Транспорт"]


def main(categories=False):
    OUT.mkdir(parents=True, exist_ok=True)
    nat = national_monthly()
    context = lambda o: {"origin": o, "national": nat[nat.index <= o]}
    models = {"seasonal_growth1": M.seasonal_growth(1), "local_sng2": M.local_sng2,
              "v2": M.v2_ensemble, **final_models(), **research_models()}
    all_rows, per_origin = [], []
    for cat in CATEGORIES if categories else CATEGORIES[:1]:
        p = load_panel(cat)
        d = B.run(p, models, B.EXTENDED_ORIGINS, horizons=[1, 2, 3, 6], ctx_fn=context)
        exact = d[d.origin.isin(B.EXACT_ORIGINS) & d.h.le(3)]
        if cat == CATEGORIES[0]:
            assert abs(B.metrics(exact, "v3")["MAE"] - 719.870192) < 1e-4
        windows = {"exact": exact, "other": d[d.origin.isin(pd.to_datetime(["2024-04-01", "2024-05-01", "2024-10-01", "2024-11-01"])) & d.h.le(3)],
                   "apr_nov": d[d.origin.ge("2024-04-01") & d.h.le(3)],
                   "all_origins": d[d.h.le(3)], "h6": d[d.h.eq(6)]}
        for name, frame in windows.items():
            rows = B.table(frame, list(models))
            rows.insert(0, "window", name); rows.insert(0, "category", cat)
            all_rows.append(rows)
        oo = B.table(d, list(models), by=["origin", "h"]); oo.insert(0, "category", cat)
        per_origin.append(oo)
        # Pairwise predictions allow honest future ensemble research without rerunning every model.
        slug = "total" if cat == CATEGORIES[0] else str(CATEGORIES.index(cat))
        d.to_csv(OUT / f"forecast_pairs_{slug}.csv.gz", index=False, compression={"method":"gzip", "mtime":0})
        print(cat, B.table(exact, list(models))[["model", "MAE", "R2_growth"]].round(3).to_string(index=False), flush=True)
    pd.concat(all_rows).to_csv(OUT / "forecast_metrics.csv", index=False)
    pd.concat(per_origin).to_csv(OUT / "forecast_by_origin.csv", index=False)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "data/raw").glob("*.csv")}
    manifest["protocol_sha256"] = hashlib.sha256((ROOT / "config/research_protocol.json").read_bytes()).hexdigest()
    (OUT / "data_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--categories", action="store_true")
    main(parser.parse_args().categories)
