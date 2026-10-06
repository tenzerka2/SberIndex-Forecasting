"""Deterministic real-municipality examples for the presentation.

Selection rules are fixed and reproducible, not hand-picked:
- stable: no detected structural shifts, exact MAE closest to the 10th percentile among no-shift rows;
- shift: at least one shift in Mar-Oct 2024, exact MAE closest to the median of that subset;
- failure: largest exact-window V3 MAE.

All three are visualized from the same forecast origin 2024-09 with h=1..3, so plots are directly
comparable. Output: outputs/real_examples.csv, REAL_EXAMPLES.md and three PNGs in figures/.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from sbx import backtest as B, early_warning as EW
from sbx.data import load_panel, national_monthly, weekly_monthly
from sbx.final import final_models
from sbx.intervals import conformal

OUT,FIG=ROOT/"outputs",ROOT/"figures"
FIG.mkdir(exist_ok=True)
ORIGIN=pd.Timestamp("2024-09-01")
ROLE_FILE={"stable":"municipality_stable.png","shift":"municipality_shift.png","failure":"municipality_failure.png"}

def select(panel):
    nat,wk=national_monthly(),weekly_monthly()
    ctx=lambda o: {"origin":o,"national":nat[nat.index<=o],"weekly":wk[wk.index<=o]}
    ex=B.run(panel,final_models(),B.EXACT_ORIGINS,ctx_fn=ctx)
    ex["ae_v3"]=(ex["v3"]-ex["y"]).abs()
    mae=ex.groupby("series")["ae_v3"].mean()
    ev=EW.events(panel.logs)
    any_ev=ev.any(axis=1)
    no_ev=np.where(~any_ev)[0]
    q10=float(mae.loc[no_ev].quantile(.10))
    stable=int((mae.loc[no_ev]-q10).abs().idxmin())

    idx={p:i for i,p in enumerate(panel.periods)}
    lo,hi=idx[pd.Timestamp("2024-03-01")],idx[pd.Timestamp("2024-10-01")]
    has_shift=ev[:,lo:hi+1].any(axis=1)
    shift_ids=np.where(has_shift)[0]
    med=float(mae.loc[shift_ids].median())
    shift=int((mae.loc[shift_ids]-med).abs().idxmin())
    failure=int(mae.idxmax())
    return {"stable":stable,"shift":shift,"failure":failure},mae,ev,ctx

def main():
    panel=load_panel()
    chosen,mae,ev,ctx=select(panel)
    # calibration + Sep forecasts for intervals
    bt=B.run(panel,final_models(),pd.date_range("2024-02-01",ORIGIN,freq="MS"),ctx_fn=ctx)
    ci=conformal(bt[["series","origin","h","target_date","y","v3"]],"v3",panel.logs,panel.periods)
    hedge=conformal(bt[["series","origin","h","target_date","y","v3_hedge"]],"v3_hedge",panel.logs,panel.periods)

    rows=[]; md=["# Реальные примеры муниципалитетов","",
        "Примеры выбраны алгоритмически до построения графиков; ручного cherry-picking нет.",""]
    for role,s in chosen.items():
        meta=panel.meta.iloc[s]
        dates=[panel.periods[t].strftime("%Y-%m") for t in np.where(ev[s])[0]]
        rule={"stable":"нет structural shifts; MAE ближе всего к 10-му перцентилю стабильных рядов",
              "shift":"есть shift в марте-октябре 2024; MAE ближе всего к медиане таких рядов",
              "failure":"максимальный MAE V3 на exact-окне"}[role]
        rows.append({"role":role,"series":s,"run_id":int(meta.run_id),"mo":meta.mo,
                     "exact_mae_v3":float(mae.loc[s]),"event_months":";".join(dates),"selection_rule":rule})
        md += [f"## {role}: {meta.mo}",f"- run_id: {int(meta.run_id)}",f"- exact MAE V3: {mae.loc[s]:.1f} ₽",
               f"- structural shifts: {', '.join(dates) if dates else 'нет'}",f"- правило выбора: {rule}",""]

        fig,ax=plt.subplots(figsize=(9,4.8))
        ax.plot(panel.periods,panel.values[s],marker="o",lw=1.8,label="факт")
        a=ci[(ci.series==s)&(ci.origin==ORIGIN)].sort_values("h")
        b=hedge[(hedge.series==s)&(hedge.origin==ORIGIN)].sort_values("h")
        if len(a):
            ax.plot(a.target_date,a.v3,marker="o",lw=2,label="V3")
            ax.fill_between(a.target_date,a.v3_lo90,a.v3_hi90,alpha=.10,label="90%")
            ax.fill_between(a.target_date,a.v3_lo80,a.v3_hi80,alpha=.18,label="80%")
        if len(b): ax.plot(b.target_date,b.v3_hedge,marker="s",lw=1.5,ls="--",label="V3-hedge")
        for t in np.where(ev[s])[0]: ax.axvline(panel.periods[t],lw=1,ls=":",alpha=.6)
        ax.axvline(ORIGIN,lw=1,alpha=.5)
        ax.set_title(f"{meta.mo} — {role}")
        ax.set_ylabel("расходы, ₽"); ax.grid(alpha=.2); ax.legend(ncol=4,fontsize=8)
        fig.tight_layout(); fig.savefig(FIG/ROLE_FILE[role],dpi=180); plt.close(fig)

    pd.DataFrame(rows).to_csv(OUT/"real_examples.csv",index=False,float_format="%.3f")
    (ROOT/"REAL_EXAMPLES.md").write_text("\n".join(md),encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__":
    main()
