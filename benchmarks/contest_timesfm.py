"""Fresh frozen TimesFM 2.5 forecasts on the exact full-Prophet evaluation origins."""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,os,sys,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
from prepare_contest_bundle import load_bundle
from contest_prophet import evaluation_origins

CONFIG=dict(max_context=32,max_horizon=12,normalize_inputs=True,per_core_batch_size=16,
    use_continuous_quantile_head=True,force_flip_invariance=True,infer_is_positive=False,fix_quantile_crossing=True)
PACKAGES=['numpy','pandas','timesfm','jax','jaxlib','flax','orbax-checkpoint']


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def context(values,t,variant):
    history=np.array(values[:,:t+1],dtype=float,copy=True)
    if variant=='raw':return history
    if variant!='yoy' or t<13:raise ValueError('YoY needs at least two historical growth observations')
    return np.log(history[:,12:])-np.log(history[:,:-12])


def project(values,t,hs,variant,point,quantiles):
    point=np.asarray(point);quantiles=np.asarray(quantiles)
    if point.ndim!=2 or quantiles.ndim!=3 or quantiles.shape[2]!=10:raise ValueError('Unexpected TimesFM output shape')
    if point.shape[0]!=len(values) or point.shape[1]<max(hs) or quantiles.shape[:2]!=point.shape:raise ValueError('Output coverage mismatch')
    idx=np.array(hs)-1
    arrays=[point[:,idx],quantiles[:,idx,1],quantiles[:,idx,9]]
    if variant=='yoy':
        if t<13 or any(t+h-12>t or t+h-12<0 for h in hs):raise ValueError('Unavailable seasonal base')
        base=values[:,[t+h-12 for h in hs]]
        arrays=[base*np.exp(a) for a in arrays]
    elif variant!='raw':raise ValueError(variant)
    arrays=[np.maximum(0,a) for a in arrays]
    if not all(np.isfinite(a).all() for a in arrays):raise ValueError('Nonfinite model output')
    if (arrays[1]>arrays[2]).any():raise ValueError('Crossed interval')
    return arrays


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--origin',type=int,required=True)
    ap.add_argument('--sample-size',type=int,default=0);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();p,_,_,bundle=load_bundle(ROOT/'data/benchmark')
    origins=list(evaluation_origins(p.periods,[1,3,6,12]));t,hs,regime=origins[args.origin]
    ids=np.arange(len(p.values)) if args.sample_size==0 else np.arange(args.sample_size)
    if args.sample_size<0 or len(ids)>len(p.values):raise ValueError('Invalid sample size')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    checkpoint=ROOT/'models/timesfm-2.5-200m-flax'
    weights={str(f.relative_to(checkpoint)):digest(f) for f in sorted(checkpoint.rglob('*')) if f.is_file()}
    if not weights:raise ValueError('Missing checkpoint')
    manifest={'status':'running','model':'TimesFM 2.5 200M Flax','model_executed':True,'origin_index':args.origin,
      'origin':str(p.periods[t]),'sample_size':len(ids),'config':CONFIG,'variants':['raw','yoy'] if regime=='main' else ['raw'],
      'data_sha256':bundle['payload_sha256'],'weights_sha256':weights,'versions':{n:importlib.metadata.version(n) for n in PACKAGES},
      'source_sha256':{n:digest(ROOT/n) for n in ['benchmarks/contest_timesfm.py','benchmarks/prepare_contest_bundle.py','benchmarks/contest_prophet.py']},
      'source_commit':os.environ.get('SOURCE_COMMIT','local'),'protocol':'Retrospective exploratory; model released after evaluation history, pretraining overlap unknown; no fine-tuning or parameter search.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    import timesfm
    model=timesfm.TimesFM_2p5_200M_flax.from_pretrained(str(checkpoint.resolve()))
    model.compile(timesfm.ForecastConfig(**CONFIG))
    values=p.values[ids];blocks=[];started=time.monotonic()
    # Predict in bounded batches and persist each batch before moving on.
    for start in range(0,len(ids),64):
        selected=ids[start:start+64];v=values[start:start+64];preds={}
        for variant in manifest['variants']:
            history=context(v,t,variant)
            point,q=model.forecast(horizon=12,inputs=[a for a in history])
            preds[variant]=project(v,t,hs,variant,point,q)
        frame=pd.concat([pd.DataFrame({'series':selected,'origin':p.periods[t],'h':h,'target_date':p.periods[t+h],
          'regime':regime,'y':v[:,t+h],**{name:a[:,j] for variant,arrays in preds.items()
          for name,a in zip([f'pred_timesfm_{variant}',f'lo80_timesfm_{variant}',f'hi80_timesfm_{variant}'],arrays)}})
          for j,h in enumerate(hs)],ignore_index=True)
        frame.to_csv(out/f'batch_{start:04}.csv.gz',index=False);blocks.append(frame)
        print(f'{p.periods[t]:%Y-%m}: {start+len(selected)}/{len(ids)} municipalities, {time.monotonic()-started:.1f}s',flush=True)
    pairs=pd.concat(blocks,ignore_index=True);pairs.to_csv(out/'pairs.csv.gz',index=False)
    manifest.update(status='completed',n_pairs=len(pairs),elapsed_seconds=time.monotonic()-started)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Completed actual TimesFM predictions:',len(pairs),flush=True)


if __name__=='__main__':main()
