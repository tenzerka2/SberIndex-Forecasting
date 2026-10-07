"""Reject missing/duplicate shards, then run the existing paired-metric aggregator."""
from pathlib import Path
import argparse,json,os,subprocess,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
from prophet_full_shard import VARIANTS,shard_ids
from prepare_contest_bundle import load_bundle
from contest_prophet import evaluation_origins


def validate_partition(manifests,variant,n,shards):
    found=[m for m in manifests if m['variant']==variant]
    if len(found)!=shards or {m['shard'] for m in found}!=set(range(shards)):
        raise ValueError(f'Missing or duplicate shards for {variant}')
    for m in found:
        if m['status']!='completed' or not m['prophet_executed'] or m['shards']!=shards or m['panel_size']!=n:
            raise ValueError('Incomplete or incompatible shard')
        if m['sample_size']!=len(shard_ids(n,m['shard'],shards)) or m['n_pairs']!=24*m['sample_size']:
            raise ValueError('Incorrect shard coverage')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/prophet_full_verified')
    ap.add_argument('--shards',type=int,default=32)
    ap.add_argument('--run-url',required=True)
    ap.add_argument('--commit',required=True)
    args=ap.parse_args();p,_,_,bundle=load_bundle(ROOT/'data/benchmark');n=len(p.values)
    if n!=2016:raise ValueError('Unexpected full panel size')
    roots=sorted(args.input.glob('full-*/manifest.json'))
    manifests=[json.loads(f.read_text()) for f in roots]
    if len(manifests)!=len(VARIANTS)*args.shards:raise ValueError('Not all expected shard manifests are present')
    for v in VARIANTS:validate_partition(manifests,v,n,args.shards)
    first=manifests[0]
    for m in manifests:
        for k in ['data_sha256','raw_data_sha256','source_sha256','versions','verified_reuse']:
            if m[k]!=first[k]:raise ValueError(f'Inconsistent {k}')
        if m['data_sha256']!=bundle['payload_sha256']:raise ValueError('Shard data differs from prepared panel')
        if {k:v for k,v in m['config'].items() if k!='prophet_variants'}!={k:v for k,v in first['config'].items() if k!='prophet_variants'}:
            raise ValueError('Protocol differs between shards')
    dirs=[];args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'shard_manifests.json').write_text(json.dumps(manifests,ensure_ascii=False,indent=2))
    for variant in VARIANTS:
        selected=[];frames=[]
        for path,m in zip(roots,manifests):
            if m['variant']!=variant:continue
            root=path.parent
            meta=pd.read_csv(root/'selected_series.csv').sort_values('series')
            ids=shard_ids(n,m['shard'],args.shards)
            np.testing.assert_array_equal(meta.series,ids)
            np.testing.assert_array_equal(meta.run_id,p.meta.iloc[ids].run_id)
            selected.append(meta)
            files=sorted(root.glob('pairs_*.csv.gz'))
            if len(files)!=11:raise ValueError('Missing per-origin file')
            f=pd.concat([pd.read_csv(x,parse_dates=['origin','target_date']) for x in files],ignore_index=True)
            expected={(int(i),p.periods[t],h,p.periods[t+h],regime) for t,hs,regime in evaluation_origins(p.periods,[1,3,6,12]) for i in ids for h in hs}
            keys=['series','origin','h','target_date','regime']
            actual=set(f[keys].itertuples(index=False,name=None))
            if f.duplicated(keys).any() or actual!=expected:raise ValueError('Exact forecast-pair coverage differs')
            yi=np.array([p.periods.get_loc(x) for x in f.target_date])
            np.testing.assert_array_equal(f.y,p.values[f.series.to_numpy(),yi])
            if not np.isfinite(f[f'pred_prophet_{variant}']).all():raise ValueError('Nonfinite forecast')
            frames.append(f)
        combined=pd.concat(frames,ignore_index=True).sort_values(['origin','series','h'])
        dest=args.output/'combined'/variant;dest.mkdir(parents=True,exist_ok=True);dirs.append(dest)
        for origin,g in combined.groupby('origin'):
            g.to_csv(dest/f'pairs_{origin:%Y%m}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        pd.concat(selected).sort_values('series').to_csv(dest/'selected_series.csv',index=False)
        m={k:v for k,v in first.items() if k not in ['shard','shards','variant','fingerprint','reused_verified_fits','new_fits_this_attempt']}
        m.update(status='completed',sample_size=n,n_pairs=len(combined),config={**first['config'],'prophet_variants':[variant]})
        (dest/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
    verified=args.output/'verified'
    subprocess.run([sys.executable,str(ROOT/'benchmarks/summarize_prophet_ci.py'),*[str(d) for d in dirs],
        '--output',str(verified),'--run-url',args.run_url,'--commit',args.commit],check=True)
    metrics=pd.read_csv(verified/'metrics.csv')
    models=['v3_hedge','v3','v4_diversified','prophet_auto','prophet_log_fourier4','prophet_log_month_dummies','seasonal_naive','national_yoy_fallback']
    lines=['# Full 2016-municipality Prophet comparison','',
        'All shards completed and all forecast pairs verified. Retrospective exploratory evaluation; not an untouched test or an organizer score.',
        '',f'Source: {args.commit}',f'Workflow: {args.run_url}','',
        '| Model | MAE h1 | MAE h3 | MAE h6 | MAE h12 |','|---|---:|---:|---:|---:|']
    for model in models:
        cells=[]
        for h in [1,3,6,12]:
            a=metrics[(metrics.model==model)&(metrics.h==h)]
            cells.append(f'{a.MAE.iloc[0]:.2f}' if len(a) else '-')
        lines.append('| '+model+' | '+' | '.join(cells)+' |')
    lines+=['','h12 is a single-origin short-history fallback, not V3/V4. All variants retained, including unstable predictions.',
        'Counts: h1=20160, h3=16128, h6=10080, h12=2016. All 2016 complete series; no random sample.',
        'The original 100-series evidence was reused only after code/data/package validation. Each reused prediction is included exactly once.']
    text='\n'.join(lines)+'\n';(args.output/'FULL_RESULTS.md').write_text(text)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as stream:stream.write(text)
    print(text)


if __name__=='__main__':main()
