"""Reproduce the selected warning methods from the prepared panel and select transparent cases."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'benchmarks'))
from prepare_contest_bundle import load_bundle
from sbx import early_warning as EW,research_warning as W

def main():
 p,_,_,_=load_bundle(ROOT/'data/benchmark');L=p.logs;features=W.features(L);events=EW.events(L)
 meta=pd.read_csv(ROOT/'outputs/graph_warning/territory_map_2024.csv.gz').set_index('series')
 out=ROOT/'outputs/submission';out.mkdir(exist_ok=True);frames=[];metrics=[];cases=[]
 for task,months,method in [('detect',range(12,22),'logistic_base'),('predict',range(14,19),'jump')]:
  labels=EW.mature_labels(L,task)
  for t in months:
   if method=='jump':score=features['jump'][:,t]
   else:
    available=EW.mature_labels(L[:,:t+1],task);train=[s for s in range(6,t) if (available[:,s]>=0).all()]
    score=W.logistic_predict(W.stack(features,list(features),train),available[:,train].T.ravel(),W.stack(features,list(features),[t]))
   alarm=W.top_budget(score,.02)
   frames.append(pd.DataFrame(dict(task=task,method=method,month=t,series=np.arange(len(L)),y=labels[:,t],score=score,alarm=alarm)))
 pairs=pd.concat(frames,ignore_index=True)
 expected=pd.read_csv(ROOT/'outputs/graph_warning/metrics.csv')
 for (task,method),g in pairs.groupby(['task','method']):
  y=g.y.astype(bool);a=g.alarm;tp=int((y&a).sum());fp=int((~y&a).sum());fn=int((y&~a).sum())
  row=dict(task=task,method=method,budget_pct=2,n_rows=len(g),true_positives=tp,false_positives=fp,false_negatives=fn,precision=tp/(tp+fp),recall=tp/(tp+fn),AP=W.average_precision(y,g.score))
  ref=expected[(expected.task==task)&(expected.method==method)&(expected.budget_pct==2)].iloc[0]
  for key in ['n_rows','true_positives','false_positives','precision','recall','AP']:np.testing.assert_allclose(row[key],ref[key],rtol=1e-10,atol=1e-12)
  metrics.append(row)
  for kind,mask in [('hit',y&a),('miss',y&~a),('false_alarm',~y&a)]:
   # Two earliest municipality-month rows with distinct municipalities, not the strongest effects.
   selected=g[mask].sort_values(['month','series']).drop_duplicates('series').head(2)
   for r in selected.itertuples():
    i=int(r.series);t=int(r.month);window=range(max(0,t-2),t+1) if task=='detect' else range(t+1,t+4)
    onsets=[u for u in window if events[i,u]];tau=onsets[0] if onsets else None
    m=meta.loc[i]
    cases.append(dict(task=task,method=method,case=kind,series=i,territory_id=int(m.territory_id),region=m.region_name,name=m.municipal_district_name,
      signal_month=str(p.periods[t].date()),event_onset=str(p.periods[tau].date()) if tau is not None else '',
      event_confirmable=str(p.periods[tau+2].date()) if tau is not None else '',
      label_confirmable=str(p.periods[t+(2 if task=='detect' else 5)].date()),
      signal_lag_months=t-tau if tau is not None else None,score=r.score,alarm=bool(r.alarm),label=int(r.y),
      previous_value=p.values[i,t-1],observed_value=p.values[i,t],monthly_change_pct=100*(p.values[i,t]/p.values[i,t-1]-1)))
 pairs.to_csv(out/'warning_pairs.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 pd.DataFrame(metrics).to_csv(out/'warning_metrics.csv',index=False)
 pd.DataFrame(cases).to_csv(out/'warning_cases.csv',index=False)
 selected_ids=sorted({r['series'] for r in cases})
 pd.DataFrame([dict(series=i,date=str(p.periods[t].date()),value=p.values[i,t],relative_log_level=EW.rel_level(L)[i,t],event=bool(events[i,t])) for i in selected_ids for t in range(24)]).to_csv(out/'case_history.csv',index=False)
 (out/'case_policy.json').write_text(json.dumps({'selection':'Two earliest rows by month then series, distinct municipalities, within each task/outcome; illustrative, not representative estimates.','labels':'Algorithm-defined persistent relative shifts, not externally confirmed economic events.','false_alarm':'No labeled onset in the task window; not proof no real-world event occurred.','score':'Ranking score, not a calibrated probability.','evaluation':'Detection labels mature at t+2; prediction labels at t+5. All training labels are mature at signal time.','prediction_method':'Jump is the strongest tested simple baseline on these already studied outcomes; not independently validated.'},indent=2))
 print(pd.DataFrame(metrics).to_string(index=False));print(pd.DataFrame(cases).to_string(index=False))

if __name__=='__main__':main()
