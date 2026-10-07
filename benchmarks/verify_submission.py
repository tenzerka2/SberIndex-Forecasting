"""Lightweight verification of saved evidence and a fresh primary-model backtest; no heavy models."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'benchmarks'))
from prepare_contest_bundle import load_bundle
from contest_prophet import evaluation_origins
from forecast_operational import predict_by_horizon
KEY=['series','origin','h','target_date','regime']

def main():
 p,nat,wk,bundle=load_bundle(ROOT/'data/benchmark');checks=[]
 expected={(i,p.periods[t],h,p.periods[t+h],regime) for t,hs,regime in evaluation_origins(p.periods,[1,3,6,12]) for i in range(2016) for h in hs}
 frames={}
 for kind in ['prophet_full','timesfm']:
  root=ROOT/'outputs'/f'{kind}_summary';path=root/'verified_pairs.csv.gz'
  proof=json.loads((root/'verification.json').read_text());field='verified_pairs_sha256' if kind=='prophet_full' else 'pairs_sha256'
  if hashlib.sha256(path.read_bytes()).hexdigest()!=proof[field]:raise ValueError('Pair checksum changed: '+kind)
  f=pd.read_csv(path,parse_dates=['origin','target_date']);frames[kind]=f
  if f.duplicated(KEY).any() or set(f[KEY].itertuples(index=False,name=None))!=expected:raise ValueError('Incorrect full coverage')
  yi=np.array([p.periods.get_loc(d) for d in f.target_date]);np.testing.assert_array_equal(f.y,p.values[f.series.to_numpy(),yi])
  for r in pd.read_csv(root/'metrics.csv').itertuples():
   g=f[(f.regime==r.regime)&(f.h==r.h)];pred=g['pred_'+r.model]
   if not np.isfinite(pred).all():raise ValueError('Incomplete predictions')
   np.testing.assert_allclose(np.abs(g.y-pred).mean(),r.MAE,rtol=1e-12)
  checks.append(kind+': exact 48,384 pairs, targets, SHA256 and every MAE verified')
 a=frames['prophet_full'].set_index(KEY).sort_index();b=frames['timesfm'].set_index(KEY).sort_index()
 for c in a:np.testing.assert_allclose(a[c],b[c],rtol=1e-12,equal_nan=True)
 for r in pd.read_csv(ROOT/'outputs/timesfm_summary/intervals.csv').itertuples():
  f=frames['timesfm'];g=f[(f.regime==r.regime)&(f.h==r.h)];lo=g['lo80_'+r.model];hi=g['hi80_'+r.model]
  np.testing.assert_allclose(((g.y>=lo)&(g.y<=hi)).mean(),r.coverage,rtol=1e-12)
  np.testing.assert_allclose((hi-lo).mean(),r.mean_width,rtol=1e-12)
 for t,hs,regime in evaluation_origins(p.periods,[1,3,6,12]):
  origin=p.periods[t];ctx={'origin':origin,'national':nat.loc[:origin],'weekly':wk.loc[:origin]}
  pred,models=predict_by_horizon(p.logs[:,:t+1],ctx)
  for h in hs:
   g=frames['prophet_full'].query('origin == @origin and h == @h').sort_values('series')
   np.testing.assert_allclose(np.exp(pred[h]),g['pred_'+models[h]],rtol=1e-10,atol=1e-8)
 checks+=['TimesFM intervals and shared baseline predictions verified','Primary model freshly recomputed at all 11 evaluation origins']
 out=ROOT/'outputs/submission';out.mkdir(exist_ok=True)
 (out/'verification.json').write_text(json.dumps({'status':'passed','checks':checks,'data_sha256':bundle['payload_sha256'],'versions':{'numpy':np.__version__,'pandas':pd.__version__},'scope':'Recompute saved metrics and primary model. Does not refit Prophet or rerun TimesFM; their actual CI evidence is separately retained.'},indent=2))
 print('\n'.join(checks))

if __name__=='__main__':main()
