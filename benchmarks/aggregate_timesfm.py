"""Validate all TimesFM origins and join onto the already verified full Prophet pairs."""
from pathlib import Path
import argparse,json,os,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'benchmarks'))
from contest_timesfm import digest
from prepare_contest_bundle import load_bundle
from contest_prophet import evaluation_origins
KEY=['series','origin','h','target_date','regime']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();p,_,_,bundle=load_bundle(ROOT/'data/benchmark')
    files=sorted(args.input.glob('tsfm-origin-*/manifest.json'))
    manifests=[json.loads(f.read_text()) for f in files]
    if len(manifests)!=11 or {m['origin_index'] for m in manifests}!=set(range(11)):raise ValueError('Missing or duplicate origins')
    frames=[];origins=list(evaluation_origins(p.periods,[1,3,6,12]))
    for file,m in zip(files,manifests):
        if m['status']!='completed' or not m['model_executed'] or m['sample_size']!=2016:raise ValueError('Incomplete run')
        for key in ['config','weights_sha256','data_sha256','source_sha256','source_commit','versions']:
            if m[key]!=manifests[0][key]:raise ValueError('Inconsistent '+key)
        if m['data_sha256']!=bundle['payload_sha256']:raise ValueError('Different input data')
        t,hs,regime=origins[m['origin_index']]
        expected={(i,p.periods[t],h,p.periods[t+h],regime) for i in range(2016) for h in hs}
        f=pd.read_csv(file.parent/'pairs.csv.gz',parse_dates=['origin','target_date'])
        if f.duplicated(KEY).any() or set(f[KEY].itertuples(index=False,name=None))!=expected or len(f)!=m['n_pairs']:raise ValueError('Pair coverage differs')
        for variant in ['raw']+(['yoy'] if regime=='main' else []):
            for prefix in ['pred_','lo80_','hi80_']:
                if not np.isfinite(f[prefix+'timesfm_'+variant]).all():raise ValueError('Missing predictions')
        frames.append(f)
    foundation=pd.concat(frames,ignore_index=True).set_index(KEY).sort_index()
    reference=pd.read_csv(args.reference/'verified/verified_pairs.csv.gz',parse_dates=['origin','target_date']).set_index(KEY).sort_index()
    provenance=json.loads((args.reference/'verified/provenance.json').read_text())
    if provenance['source_commit']!='b61185ef2a01714982e57c37dcd5fd33b2a8dc5c':raise ValueError('Unexpected Prophet source commit')
    if any(m['data_sha256']!=bundle['payload_sha256'] for m in provenance['manifests']):raise ValueError('Reference data differs')
    if not reference.index.equals(foundation.index):raise ValueError('Reference forecast pairs differ')
    np.testing.assert_array_equal(reference.y,foundation.y)
    merged=reference.join(foundation.drop(columns='y')).reset_index()
    yi=np.array([p.periods.get_loc(x) for x in merged.target_date]);np.testing.assert_array_equal(merged.y,p.values[merged.series.to_numpy(),yi])
    metrics=[];intervals=[];comparisons=[]
    for (regime,h),g in merged.groupby(['regime','h']):
        for col in [c for c in g if c.startswith('pred_') and not g[c].isna().all()]:
            y=g.y.to_numpy();pred=g[col].to_numpy()
            if not np.isfinite(pred).all():raise ValueError('Partial model coverage')
            metrics.append(dict(regime=regime,h=h,model=col[5:],n_pairs=len(g),MAE=np.abs(y-pred).mean(),R2_level=1-((y-pred)**2).sum()/((y-y.mean())**2).sum()))
            if col.startswith('pred_timesfm_'):
                name=col[5:];lo=g['lo80_'+name].to_numpy();hi=g['hi80_'+name].to_numpy()
                if (lo>hi).any():raise ValueError('Crossed intervals')
                score=hi-lo+10*np.maximum(lo-y,0)+10*np.maximum(y-hi,0)
                intervals.append(dict(regime=regime,h=h,model=name,n_pairs=len(g),nominal_coverage=.8,coverage=((y>=lo)&(y<=hi)).mean(),mean_width=(hi-lo).mean(),interval_score=score.mean()))
                candidate='pred_v3_hedge' if regime=='main' else 'pred_national_yoy_fallback'
                e=(g[candidate]-g.y).abs();b=(g[col]-g.y).abs();diff=e-b
                comparisons.append(dict(regime=regime,h=h,candidate=candidate[5:],baseline=name,MAE_reduction_pct=100*(1-e.mean()/b.mean()),municipalities_better_pct=100*(diff.groupby(g.series).mean()<0).mean(),origins_better=int((diff.groupby(g.origin).mean()<0).sum()),origins_total=g.origin.nunique()))
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('metrics',metrics),('intervals',intervals),('comparisons',comparisons)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    merged.to_csv(out/'verified_pairs.csv.gz',index=False)
    (out/'provenance.json').write_text(json.dumps({'manifests':manifests,'prophet_provenance':provenance,'reference_pairs_sha256':digest(args.reference/'verified/verified_pairs.csv.gz'),'workflow_url':os.environ.get('RUN_URL'),'notes':['Retrospective exploratory. Foundation checkpoint released after evaluation dates; overlap with pretraining data cannot be ruled out.','Raw and log-YoY protocols fixed before this run; no fine tuning. Short contexts and one h12 origin.','Intervals are nominal 80%, evaluated without recalibration.']} ,indent=2))
    table=pd.DataFrame(metrics).pivot(index='model',columns='h',values='MAE').round(2)
    report='# TimesFM 2.5 versus full-panel Prophet and V3 Hedge\n\nRetrospective exploratory comparison; not an untouched test.\n\n'+table.to_string()+'\n\n80% interval diagnostics:\n'+pd.DataFrame(intervals).round(4).to_string(index=False)+'\n\nPaired comparisons:\n'+pd.DataFrame(comparisons).round(3).to_string(index=False)+'\n'
    (out/'RESULTS.txt').write_text(report);print(report)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as f:f.write('```text\n'+report+'\n```\n')

if __name__=='__main__':main()
