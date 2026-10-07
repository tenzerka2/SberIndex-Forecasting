"""Full-panel Prophet in independent resumable shards, using the frozen fitted model."""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
from contest_prophet import evaluation_origins,fit_prophet,final_models,v4_models,summarize
from prepare_contest_bundle import load_bundle

VARIANTS=['auto','log_fourier4','log_month_dummies']
SOURCE_FILES=['benchmarks/contest_prophet.py','benchmarks/prepare_contest_bundle.py',
    'src/sbx/models.py','src/sbx/candidates.py','src/sbx/final.py','src/sbx/data.py']


def shard_ids(n,shard,shards):
    if not 1<=shards<=n or not 0<=shard<shards:raise ValueError('Invalid shard bounds')
    return np.arange(n)[shard::shards]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_cache(root,p,bundle,versions,cfg):
    """Only use the published actual-Prophet run when data, fit code and versions match."""
    provenance=json.loads((root/'provenance.json').read_text())
    for m in provenance['manifests']:
        if m['status']!='completed' or not m['prophet_executed']:raise ValueError('Unverified cache')
        if m['data_sha256']!=bundle['payload_sha256'] or m['raw_data_sha256']!=bundle['raw_sha256']:
            raise ValueError('Cached data differs')
        if m['versions']!=versions:raise ValueError('Cached package versions differ')
        for name,digest in m['source_sha256'].items():
            if sha(ROOT/name)!=digest:raise ValueError(f'Cached model source differs: {name}')
        for key in ['seed','horizons','min_history','prediction_floor_rub']:
            if m['config'][key]!=cfg[key]:raise ValueError('Cached protocol differs')
    df=pd.read_csv(root/'verified_pairs.csv.gz',parse_dates=['origin','target_date'])
    key=['series','origin','h','target_date','regime']
    if df.duplicated(key).any():raise ValueError('Duplicate cached pairs')
    valid=set()
    for t,hs,regime in evaluation_origins(p.periods,cfg['horizons'],cfg['min_history']):
        valid.update((p.periods[t],h,p.periods[t+h],regime) for h in hs)
    for row in df.itertuples():
        if not 0<=row.series<len(p.values) or (row.origin,row.h,row.target_date,row.regime) not in valid:
            raise ValueError('Invalid cached pair')
        if row.y!=p.values[row.series,p.periods.get_loc(row.target_date)]:raise ValueError('Cached target differs')
    for variant in VARIANTS:
        if not np.isfinite(df[f'pred_prophet_{variant}']).all():raise ValueError('Incomplete cached predictions')
    return df,{'run_url':provenance['workflow_run_url'],'source_commit':provenance['source_commit'],
        'pairs_sha256':sha(root/'verified_pairs.csv.gz'),'provenance_sha256':sha(root/'provenance.json')}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variant',choices=VARIANTS,required=True)
    ap.add_argument('--shard',type=int,required=True)
    ap.add_argument('--shards',type=int,default=32)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--no-cache',action='store_true')
    args=ap.parse_args()
    if args.workers<1:ap.error('workers must be positive')
    p,nat,wk,bundle=load_bundle(ROOT/'data/benchmark')
    ids=shard_ids(len(p.values),args.shard,args.shards)
    cfg=json.loads((ROOT/'config/contest_prophet.json').read_text())
    cfg.update(sample_size=0,prophet_variants=[args.variant])
    versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','prophet','cmdstanpy']}
    cached,cache_info=(pd.DataFrame(),None) if args.no_cache else verified_cache(ROOT/'outputs/prophet_verified',p,bundle,versions,cfg)
    manifest={'status':'running','prophet_executed':True,'config':cfg,'sample_size':len(ids),
        'panel_size':len(p.values),'shard':args.shard,'shards':args.shards,'variant':args.variant,
        'data_sha256':bundle['payload_sha256'],'raw_data_sha256':bundle['raw_sha256'],
        'source_sha256':{n:sha(ROOT/n) for n in SOURCE_FILES+['benchmarks/prophet_full_shard.py']},
        'versions':versions,'verified_reuse':cache_info,
        'protocol':'Full panel, retrospective exploratory; same history already studied; no untouched holdout',
        'category':'Все категории','selection':'Complete positive 24-month series; retrospective selection',
        'exogenous':'Current-vintage national observations truncated by origin, release vintages unknown'}
    fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    manifest['fingerprint']=fingerprint
    out=args.output;mp=out/'manifest.json'
    if mp.exists():
        if not args.resume:raise ValueError('Output exists; use --resume or a fresh output')
        if json.loads(mp.read_text())['fingerprint']!=fingerprint:raise ValueError('Resume fingerprint differs')
    out.mkdir(parents=True,exist_ok=True)
    mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    selected=p.meta.iloc[ids].copy();selected.insert(0,'series',ids)
    selected.to_csv(out/'selected_series.csv',index=False)
    models={**final_models(),**{k:v for k,v in v4_models().items() if k=='v4_diversified'}}
    blocks=[];reused=0;new_fits=0;started=time.monotonic()
    for t,hs,regime in evaluation_origins(p.periods,cfg['horizons'],cfg['min_history']):
        origin=p.periods[t];history=p.logs[:,:t+1].copy();history.setflags(write=False)
        ctx={'origin':origin,'national':nat.loc[:origin],'weekly':wk.loc[:origin]}
        preds={'seasonal_naive':{h:history[:,t+h-12] for h in hs}}
        if regime=='main':preds.update({name:fn(history,hs,ctx) for name,fn in models.items()})
        else:
            spend=ctx['national'].nat_spend.dropna()
            growth=np.log(spend.iloc[-1]/spend.loc[spend.index[-1]-pd.DateOffset(years=1)])
            preds['national_yoy_fallback']={h:history[:,t+h-12]+growth for h in hs}
        block=pd.concat([pd.DataFrame({'series':ids,'origin':origin,'h':h,'target_date':p.periods[t+h],
            'regime':regime,'y':p.values[ids,t+h],
            **{'pred_'+name:np.maximum(0,np.exp(pred[h][ids])) for name,pred in preds.items()}}) for h in hs],ignore_index=True)
        fitted={};checkpoint=out/f'fits_{origin:%Y%m}.jsonl'
        if not cached.empty:
            subset=cached[cached.origin.eq(origin)&cached.series.isin(ids)]
            for i,g in subset.groupby('series'):
                if sorted(g.h.tolist())!=sorted(hs):raise ValueError('Cache horizon mismatch')
                fitted[int(i)]=g.set_index('h').loc[hs,f'pred_prophet_{args.variant}'].to_numpy().tolist()
                reused+=1
        if checkpoint.exists():
            for line in checkpoint.read_text().splitlines():
                r=json.loads(line)
                if r['fingerprint']!=fingerprint or r['hs']!=hs or r['series'] not in ids:
                    raise ValueError('Invalid checkpoint')
                a=np.asarray(r['predictions'],float)
                if len(a)!=len(hs) or not np.isfinite(a).all() or (a<0).any():raise ValueError('Bad cached fit')
                if r['series'] in fitted:np.testing.assert_allclose(a,fitted[r['series']],rtol=1e-12)
                fitted[r['series']]=a.tolist()
        tasks=[(int(i),list(p.periods[:t+1]),p.values[i,:t+1],hs,args.variant,cfg['seed']) for i in ids if i not in fitted]
        print(f'{args.variant} shard {args.shard} {origin:%Y-%m}: {len(fitted)}/{len(ids)} cached, {len(tasks)} new',flush=True)
        with ProcessPoolExecutor(args.workers) as pool,checkpoint.open('a') as stream:
            futures={pool.submit(fit_prophet,task):task[0] for task in tasks}
            for future in as_completed(futures):
                i,v=future.result();v=np.asarray(v).tolist()
                stream.write(json.dumps({'fingerprint':fingerprint,'series':i,'hs':hs,'predictions':v})+'\n');stream.flush()
                fitted[i]=v;new_fits+=1
                if len(fitted)%10==0 or len(fitted)==len(ids):
                    print(f'{origin:%Y-%m}: {len(fitted)}/{len(ids)}, total elapsed {time.monotonic()-started:.0f}s',flush=True)
        if set(fitted)!=set(ids):raise ValueError('Incomplete fit coverage')
        mapping={(i,h):float(v) for i,values in fitted.items() for h,v in zip(hs,values)}
        block[f'pred_prophet_{args.variant}']=[mapping[i,h] for i,h in zip(block.series,block.h)]
        block.to_csv(out/f'pairs_{origin:%Y%m}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        blocks.append(block)
    pairs=pd.concat(blocks,ignore_index=True);metrics,by_origin=summarize(pairs)
    metrics.to_csv(out/'metrics.csv',index=False);by_origin.to_csv(out/'metrics_by_origin.csv',index=False)
    manifest.update(status='completed',n_pairs=len(pairs),reused_verified_fits=reused,new_fits_this_attempt=new_fits)
    mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(f'Completed: {len(pairs)} pairs; reused {reused}; fitted {new_fits}',flush=True)


if __name__=='__main__':main()
