"""Second explicitly exploratory phase: national seasonal shape vs municipal base anomalies."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.data import ROOT, load_panel, national_monthly
from sbx import backtest as B, models as M
from sbx.final import final_models
from sbx.candidates import national_shape, national_shape_full

OUT = ROOT / "outputs/research"


def main():
    nat = national_monthly()
    ctx = lambda o: {"origin":o,"national":nat.loc[:o]}
    models = {**final_models(), "national_shape_50":national_shape,
              "national_shape_100":national_shape_full,
              "national_shape_50_feedback":M.error_feedback(base=national_shape)}
    rows=[]
    cats = ["Все категории", "Здоровье", "Маркетплейсы", "Общественное питание", "Продовольствие", "Транспорт"]
    for cat in cats:
        d=B.run(load_panel(cat),models,B.EXTENDED_ORIGINS,horizons=[1,2,3,6],ctx_fn=ctx)
        for name,m in {"exact":d.origin.isin(B.EXACT_ORIGINS)&d.h.le(3),
                       "other":d.origin.isin(pd.to_datetime(["2024-04-01","2024-05-01","2024-10-01","2024-11-01"]))&d.h.le(3),
                       "apr_nov":d.origin.ge("2024-04-01")&d.h.le(3),
                       "all_origins":d.h.le(3),"h6":d.h.eq(6)}.items():
            z=B.table(d[m],list(models)); z.insert(0,"window",name); z.insert(0,"category",cat); rows.append(z)
        if cat==cats[0]:
            d.to_csv(OUT/"national_shape_pairs.csv.gz",index=False,compression={"method":"gzip","mtime":0})
        print(cat,B.table(d[d.origin.isin(B.EXACT_ORIGINS)&d.h.le(3)],list(models))[["model","MAE"]].round(3).to_string(index=False),flush=True)
    pd.concat(rows).to_csv(OUT/"national_shape_metrics.csv",index=False)


if __name__ == "__main__": main()
