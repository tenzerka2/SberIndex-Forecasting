"""Final exploratory phase: a fixed diversified common trend and error feedback."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.data import ROOT, load_panel, national_monthly
from sbx import backtest as B
from sbx.final import final_models
from sbx.candidates import v4_models
OUT = ROOT / "outputs/research"


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    nat=national_monthly(); ctx=lambda o:{"origin":o,"national":nat.loc[:o]}
    models={**final_models(),**v4_models()}; rows=[]; by_origin=[]
    cats=["Все категории","Здоровье","Маркетплейсы","Общественное питание","Продовольствие","Транспорт"]
    for j,cat in enumerate(cats):
        d=B.run(load_panel(cat),models,B.EXTENDED_ORIGINS,horizons=[1,2,3,6],ctx_fn=ctx)
        windows={"exact":d.origin.isin(B.EXACT_ORIGINS)&d.h.le(3),
                 "other":d.origin.isin(pd.to_datetime(["2024-04-01","2024-05-01","2024-10-01","2024-11-01"]))&d.h.le(3),
                 "apr_nov":d.origin.ge("2024-04-01")&d.h.le(3),"all_origins":d.h.le(3),"h6":d.h.eq(6)}
        for name,m in windows.items():
            z=B.table(d[m],list(models)); z.insert(0,"window",name); z.insert(0,"category",cat); rows.append(z)
        z=B.table(d,list(models),by=["origin","h"]);z.insert(0,"category",cat);by_origin.append(z)
        d.to_csv(OUT/f"v4_pairs_{j}.csv.gz",index=False,compression={"method":"gzip","mtime":0})
        print(cat,B.table(d[windows['exact']],list(models))[["model","MAE"]].round(3).to_string(index=False),flush=True)
    pd.concat(rows).to_csv(OUT/"v4_metrics.csv",index=False)
    pd.concat(by_origin).to_csv(OUT/"v4_by_origin.csv",index=False)


if __name__=="__main__": main()
