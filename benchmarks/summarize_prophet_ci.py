"""Validate and aggregate completed CI artifacts on strictly identical forecast pairs."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd

KEY=['series','origin','h','target_date','regime']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directories',nargs='+',type=Path)
    ap.add_argument('--output',type=Path,default=Path('outputs/prophet_verified'))
    ap.add_argument('--run-url',default='')
    ap.add_argument('--commit',default='')
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    frames=[]; manifests=[]; input_hashes=[]; selected_frames=[]
    for root in args.directories:
        files=sorted(root.glob('pairs_*.csv.gz'))
        manifest=json.loads((root/'manifest.json').read_text())
        if manifest['status']!='completed' or not manifest['prophet_executed']:
            raise ValueError(f'Incomplete or non-Prophet result: {root}')
        if len(files)!=11:
            raise ValueError(f'Missing origins in {root}')
        frame=pd.concat([pd.read_csv(f,parse_dates=['origin','target_date']) for f in files],ignore_index=True)
        selected=pd.read_csv(root/'selected_series.csv').sort_values('series').reset_index(drop=True)
        if selected.series.duplicated().any() or set(selected.series)!=set(frame.series):
            raise ValueError('Invalid selected-series mapping')
        selected_frames.append(selected)
        if frame.duplicated(KEY).any() or len(frame)!=manifest['n_pairs']:
            raise ValueError('Invalid pair coverage')
        expected={1:10,3:8,6:5,12:1}
        if frame.groupby('h').size().to_dict()!={h:n*manifest['sample_size'] for h,n in expected.items()}:
            raise ValueError('Incomplete horizon coverage')
        if not np.isfinite(frame.y).all():
            raise ValueError('Non-finite observed values')
        for variant in manifest['config']['prophet_variants']:
            column=f'pred_prophet_{variant}'
            if column not in frame or not np.isfinite(frame[column]).all():
                raise ValueError(f'Incomplete Prophet variant: {variant}')
        input_hashes.append({f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                             for f in files+[root/'manifest.json',root/'selected_series.csv']})
        frames.append(frame.set_index(KEY).sort_index());manifests.append(manifest)
    first=manifests[0]
    for selected in selected_frames[1:]:
        pd.testing.assert_frame_equal(selected_frames[0],selected)
    for manifest in manifests[1:]:
        if {k:v for k,v in manifest['config'].items() if k!='prophet_variants'}!={k:v for k,v in first['config'].items() if k!='prophet_variants'}:
            raise ValueError('Inconsistent protocol configuration')
        for k in ['sample_size','data_sha256','raw_data_sha256','source_sha256','versions']:
            if manifest[k]!=first[k]:
                raise ValueError(f'Inconsistent {k}')
    merged=frames[0].copy()
    for frame in frames[1:]:
        if not merged.index.equals(frame.index):
            raise ValueError('Forecast pairs differ across variants')
        for c in frame:
            if c in merged:
                np.testing.assert_allclose(merged[c],frame[c],rtol=1e-12,equal_nan=True)
            else:
                merged[c]=frame[c]
    merged=merged.reset_index()
    models=[c for c in merged if c.startswith('pred_')]
    metrics=[]; comparisons=[]
    for (regime,h), group in merged.groupby(['regime','h']):
        available=[c for c in models if not group[c].isna().all()]
        for model in available:
            if not np.isfinite(group[model]).all(): raise ValueError('Missing model predictions')
            y=group.y.to_numpy(); pred=group[model].to_numpy()
            ybase=group.pred_seasonal_naive.to_numpy()
            growth=np.log(y/ybase)
            growth_r2=(1-((growth-np.log(pred/ybase))**2).sum()/((growth-growth.mean())**2).sum()) if (pred>0).all() else np.nan
            metrics.append({'regime':regime,'h':h,'model':model[5:],'n_pairs':len(group),
                'n_origins':group.origin.nunique(),'MAE':np.abs(y-pred).mean(),
                'R2_level':1-((y-pred)**2).sum()/((y-y.mean())**2).sum(),
                'R2_log_yoy':growth_r2,
                'wMAPE_pct':100*np.abs(y-pred).sum()/y.sum()})
        for baseline in [c for c in available if c.startswith('pred_prophet')]:
            for candidate in [c for c in available if not c.startswith('pred_prophet')]:
                canderr=(group[candidate]-group.y).abs(); baseerr=(group[baseline]-group.y).abs()
                diff=canderr-baseerr
                monthly=diff.groupby(group.target_date).mean()
                byseries=diff.groupby(group.series).mean()
                rng=np.random.default_rng(0)
                # Descriptive cluster bootstrap; 5-10 months do not support strong significance claims.
                draws=rng.choice(monthly.to_numpy(),size=(2000,len(monthly)),replace=True).mean(axis=1)
                comparisons.append({'regime':regime,'h':h,'model':candidate[5:],'baseline':baseline[5:],
                    'MAE_reduction_pct':100*(1-canderr.mean()/baseerr.mean()),
                    'MAE_difference_rub':diff.mean(),'months_better':int((monthly<0).sum()),
                    'months_total':len(monthly),'municipalities_better_pct':100*(byseries<0).mean(),
                    'descriptive_month_bootstrap_lo':np.quantile(draws,.025),
                    'descriptive_month_bootstrap_hi':np.quantile(draws,.975)})
    pd.DataFrame(metrics).to_csv(args.output/'metrics.csv',index=False)
    pd.DataFrame(comparisons).to_csv(args.output/'comparisons.csv',index=False)
    merged.to_csv(args.output/'verified_pairs.csv.gz',index=False)
    selected_frames[0].to_csv(args.output/'selected_series.csv',index=False)
    provenance={'workflow_run_url':args.run_url,'source_commit':args.commit,
        'input_sha256':input_hashes,'manifests':manifests,'notes':['Retrospective exploratory evaluation; data already inspected during development.',
        'Prepared data verified against original exports. National signals are current-vintage.',
        'Bootstrap intervals descriptive only: few target months and model selection on related history.',
        'R2_log_yoy compares log(y/y_previous_year) and log(pred/y_previous_year); undefined when predictions contain zero.',
        'h12 fallback is a separate method at a single forecast origin, not V3/V4.']}
    (args.output/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    print(pd.DataFrame(metrics).round(3).to_string(index=False))
    print(pd.DataFrame(comparisons).query("model == 'v3_hedge'").round(3).to_string(index=False))

if __name__=='__main__':main()
