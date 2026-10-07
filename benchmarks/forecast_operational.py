"""Export a clearly dated forecast with a fixed, transparent short-history fallback."""
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'benchmarks'))
from sbx.final import final_models
from prepare_contest_bundle import load_bundle


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--origin',default='2024-12-01')
    ap.add_argument('--prepared',type=Path,default=ROOT/'data/benchmark')
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/operational')
    ap.add_argument('--territory-map',type=Path,help='Verified official-ID mapping; defaults to the origin-year map if available')
    args=ap.parse_args();p,nat,wk,manifest=load_bundle(args.prepared)
    origin=pd.Timestamp(args.origin)
    if origin not in p.periods:
        raise SystemExit('Origin must be an observed municipal month; no silently invented current history')
    t=p.periods.get_loc(origin);history=p.logs[:,:t+1]
    map_path=args.territory_map or ROOT/f'outputs/graph_warning/territory_map_{origin.year}.csv.gz'
    metadata=None
    if map_path.exists():
        metadata=pd.read_csv(map_path,dtype={'oktmo':str}).sort_values('series').reset_index(drop=True)
        if len(metadata)!=len(p.meta) or metadata.territory_id.duplicated().any():
            raise SystemExit('Official-ID mapping is incomplete or ambiguous')
        np.testing.assert_array_equal(metadata.run_id.to_numpy(),p.meta.run_id.to_numpy())
        np.testing.assert_array_equal(metadata.mo.to_numpy(),p.meta.mo.to_numpy())
        if not ((metadata.year_from<=origin.year)&(metadata.year_to>origin.year)).all():
            raise SystemExit('Territory metadata is not valid in the requested origin year')
    elif args.territory_map:
        raise SystemExit(f'Territory map does not exist: {map_path}')
    if history.shape[1]<12:raise SystemExit('At least 12 monthly observations are required')
    hs=[1,3,6,12];ctx={'origin':origin,'national':nat.loc[:origin],'weekly':wk.loc[:origin]}
    if history.shape[1]>=14:
        model='v3_hedge';pred=final_models()[model](history,hs,ctx)
    else:
        model='national_yoy_fallback'
        spend=ctx['national'].nat_spend.dropna()
        growth=np.log(spend.iloc[-1]/spend.loc[spend.index[-1]-pd.DateOffset(years=1)])
        pred={h:history[:,t+h-12]+growth for h in hs}
    rows=[]
    for h in hs:
        rows.append(pd.DataFrame({'run_id':p.meta.run_id,'municipality_name':p.meta.mo,
            'ambiguous_name':p.meta.homonym,'origin':origin,'h':h,
            'target_date':origin+pd.DateOffset(months=h),'forecast_rub':np.exp(pred[h]),
            'model':model,'historical_demo':True,
            'horizon_validation':'single-origin fallback only; V3-hedge h12 not backtested' if h==12 else 'retrospective 2024 rolling evaluation'}))
    result=pd.concat(rows,ignore_index=True)
    if metadata is not None:
        cols=['run_id','territory_id','region_code','region_name','oktmo','municipal_district_name']
        result=result.merge(metadata[cols].rename(columns={'municipal_district_name':'official_municipality_name'}),on='run_id',validate='many_to_one')
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/f'forecast_{origin:%Y%m}.csv.gz';result.to_csv(path,index=False)
    (args.output/'manifest.json').write_text(json.dumps({'origin':str(origin.date()),'model':model,
        'n_forecasts':len(result),'official_ids_attached':metadata is not None,
        'metadata_year':origin.year if metadata is not None else None,
        'inputs':manifest,'warning':'Historical example, not a forecast from October 2026. No calibrated uncertainty or guaranteed shock prevention claimed.'},ensure_ascii=False,indent=2))
    print(f'{len(result)} historical forecasts saved to {path}')

if __name__=='__main__':main()
