"""Frozen exploratory transport graph ablation on official-ID-aligned observations."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sbx.data import load_panel
from sbx import early_warning as EW,research_warning as W
from sbx.municipal_graph import match_official_ids,transport_neighbors,graph_features,permuted_neighbors


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=ROOT/'data/raw/hackathon')
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/graph_warning')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'config/graph_warning.json').read_text())
    p=load_panel();L=p.logs;n,T=L.shape
    consumption=pd.read_parquet(args.data/'consumption.parquet')
    mapping=match_official_ids(p,consumption)
    mapping.to_csv(args.output/'identity_map.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    connections=pd.read_parquet(args.data/'connection.parquet')
    neighbors,weights,km=transport_neighbors(mapping.territory_id,connections,cfg['neighbors'])
    mask=weights.ravel()>0
    pd.DataFrame({'territory_id':np.repeat(mapping.territory_id,cfg['neighbors']),
        'neighbor_id':mapping.territory_id.to_numpy()[neighbors].ravel(),
        'distance_km':km.ravel(),'weight':weights.ravel()}).loc[mask].to_csv(
            args.output/'neighbors.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    base=W.features(L)
    gf=graph_features(base,neighbors,weights,cfg['neighbor_features'])
    pn,pw=permuted_neighbors(neighbors,weights,cfg['permuted_graph_seed'])
    pf=graph_features(base,pn,pw,cfg['neighbor_features'])
    # Check prefix stability for the observed-value features. This does NOT make the static graph historical.
    prefix=graph_features(W.features(L[:,:18]),neighbors,weights,cfg['neighbor_features'])
    for k in gf:np.testing.assert_array_equal(gf[k][:,:18],prefix[k])
    frames=[]
    for task,ts in [('detect',cfg['detection_month_indices']),('predict',cfg['prediction_month_indices'])]:
        full=EW.mature_labels(L,task)
        for t in ts:
            available=EW.mature_labels(L[:,:t+1],task)
            train=[s for s in range(6,t) if (available[:,s]>=0).all()]
            yt=available[:,train].T.ravel();y=full[:,t]
            assert (y>=0).all() and (yt>=0).all()
            scores={'jump':base['jump'][:,t],'neighbor_jump':gf['neighbor_jump'][:,t]}
            for name,f in [('logistic_base',base),('logistic_graph',{**base,**gf}),('logistic_permuted_graph',{**base,**pf})]:
                scores[name]=W.logistic_predict(W.stack(f,list(f),train),yt,W.stack(f,list(f),[t]),cfg['logistic_penalty'])
            for name,s in scores.items():
                frame=pd.DataFrame({'task':task,'month':t,'origin':p.periods[t],
                    'series':np.arange(n),'territory_id':mapping.territory_id,'method':name,'y':y,'score':s})
                for budget in cfg['budgets_pct']:frame[f'alarm_{budget}']=W.top_budget(s,budget/100)
                frames.append(frame)
            print(task,p.periods[t].date(),int(y.sum()),flush=True)
    pairs=pd.concat(frames,ignore_index=True);rows=[];monthly=[]
    for (task,name),g in pairs.groupby(['task','method']):
        y=g.y.to_numpy(bool);s=g.score.to_numpy()
        for budget in cfg['budgets_pct']:
            a=g[f'alarm_{budget}'].to_numpy(bool);tp=int((a&y).sum());fp=int((a&~y).sum())
            row={'task':task,'method':name,'budget_pct':budget,'n_rows':len(g),'positives':int(y.sum()),
                'AP':W.average_precision(y,s),'base_rate':float(y.mean()),'precision':tp/max(int(a.sum()),1),
                'recall':tp/max(int(y.sum()),1),'true_positives':tp,'false_positives':fp,'false_alarms_per_100':100*fp/len(y)}
            if task=='predict':
                events=EW.events(L);alarm=np.zeros_like(events);lo,hi=g.month.min(),g.month.max()
                for t,m in g.groupby('month'):alarm[m.series.to_numpy(),t]=m[f'alarm_{budget}'].to_numpy()
                total=hits=0;leads=[]
                for i,tau in zip(*np.where(events)):
                    if tau-3>=lo and tau-1<=hi:
                        total+=1;hit=np.flatnonzero(alarm[i,tau-3:tau])
                        if len(hit):hits+=1;leads.append(int(3-hit[0]))
                row.update(events_with_full_warning_window=total,events_warned=hits,
                    event_recall=hits/max(total,1),median_lead_months=float(np.median(leads)) if leads else None)
            rows.append(row)
        for t,m in g.groupby('month'):
            a=m.alarm_2.to_numpy(bool);y0=m.y.to_numpy(bool)
            monthly.append({'task':task,'method':name,'month':t,'origin':m.origin.iloc[0],
                'AP':W.average_precision(y0,m.score),'precision_at_2pct':float(y0[a].mean()),
                'true_positives_at_2pct':int((y0&a).sum()),'positives':int(y0.sum())})
    result=pd.DataFrame(rows)
    # Confirm that this experiment has not accidentally changed the comparison task.
    previous=ROOT/'outputs/research/warning_metrics.csv'
    reproduced=False
    if previous.exists():
        old=pd.read_csv(previous).set_index(['task','method','budget_pct'])
        for row in rows:
            if row['method'] in ['jump','logistic_base']:
                ref=old.loc[(row['task'],row['method'],row['budget_pct'])]
                for field in ['AP','precision','recall']:np.testing.assert_allclose(row[field],ref[field],rtol=1e-10,atol=1e-12)
        reproduced=True
    result.to_csv(args.output/'metrics.csv',index=False)
    pd.DataFrame(monthly).to_csv(args.output/'metrics_by_month.csv',index=False)
    pairs.to_csv(args.output/'pairs.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    import pyarrow
    sources=[args.data/x for x in ['consumption.parquet','connection.parquet','market_access.parquet','dataset_description.pdf']]
    code=[ROOT/x for x in ['config/graph_warning.json','src/sbx/municipal_graph.py','benchmarks/graph_warning.py','src/sbx/research_warning.py','src/sbx/early_warning.py']]
    audit={'status':'completed','protocol':cfg,'n_official_ids_all':int(consumption.territory_id.nunique()),
        'n_official_rows':len(consumption),'n_exactly_matched_complete_series':n,
        'legacy_homonym_series_resolved_to_ids':int(mapping.homonym.sum()),
        'n_series_without_highway_neighbors':int((weights.sum(axis=1)==0).sum()),
        'n_directed_neighbor_edges':int(mask.sum()),'observed_feature_prefix_check':'passed',
        'legacy_baseline_metrics_reproduced':reproduced,
        'versions':{'numpy':np.__version__,'pandas':pd.__version__,'pyarrow':pyarrow.__version__},
        'input_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in code},
        'data_source':'SberIndex; user-supplied hackathonlicence.zip; received 2026-10-07',
        'license_as_stated_in_description':'CC BY-SA 4.0',
        'notes':['Full-history signatures recover identity only, not predictive features.',
            'All source observations already inspected; no independent holdout.',
            '2024-12-31 transport graph availability at earlier origins is NOT established.',
            'A single graph permutation is a diagnostic control, not a statistical significance test.',
            'Market access is not used: its historical publication timing is unknown.']}
    (args.output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(result[result.budget_pct.eq(2)].round(4).to_string(index=False))


if __name__=='__main__':main()
