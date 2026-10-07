"""Reproducible monthly comparison; run --help. No historical Prophet caches are used."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sbx.data import load_panel, national_monthly, weekly_monthly, find
from sbx.final import final_models
from sbx.candidates import v4_models


def evaluation_origins(periods, horizons, min_history=14):
    """Main history >=14; h12 has one separately labelled 12-month-history origin."""
    for t in range(min_history - 1, len(periods) - 1):
        hs = [h for h in horizons if t + h < len(periods)]
        if hs:
            yield t, hs, 'main'
    if 12 in horizons and len(periods) >= 24:
        yield 11, [12], 'short_history_fallback'


def fit_prophet(task):
    from prophet import Prophet
    series, ds, y, hs, variant, seed = task
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    logging.getLogger('prophet').setLevel(logging.ERROR)
    train = pd.DataFrame({'ds': ds, 'y': y})
    future = pd.DataFrame({'ds': [ds[-1] + pd.DateOffset(months=h) for h in hs]})
    kw = dict(weekly_seasonality=False, daily_seasonality=False, uncertainty_samples=0)
    if variant == 'auto':
        kw['yearly_seasonality'] = 'auto'
    elif variant == 'log_fourier4':
        kw.update(yearly_seasonality=4, changepoint_prior_scale=0.01)
        train['y'] = np.log(y)
    elif variant == 'log_month_dummies':
        kw.update(yearly_seasonality=False, changepoint_prior_scale=0.01)
        train['y'] = np.log(y)
    else:
        raise ValueError(variant)
    model = Prophet(**kw)
    if variant == 'log_month_dummies':
        # December is the reference month. Monthly regressors avoid daily interpolation.
        for month in range(1, 12):
            name = f'month_{month}'
            train[name] = (train.ds.dt.month == month).astype(float)
            future[name] = (future.ds.dt.month == month).astype(float)
            model.add_regressor(name, prior_scale=0.1, standardize=False)
    model.fit(train, seed=seed)
    pred = model.predict(future).yhat.to_numpy()
    if variant != 'auto':
        pred = np.exp(pred)
    if not np.isfinite(pred).all():
        raise ValueError(f'Nonfinite Prophet forecast: {series}, {variant}')
    # Same nonnegative-spending constraint for every model, recorded in protocol.
    return series, np.maximum(0.0, pred)


def scores(frame, model):
    y, pred = frame.y.to_numpy(), frame[model].to_numpy()
    if not np.isfinite(pred).all():
        raise ValueError(f'Incomplete predictions for {model}; refusing selective comparison')
    denom = ((y - y.mean()) ** 2).sum()
    return {'MAE': float(np.abs(y-pred).mean()),
            'R2_level': float(1 - ((y-pred)**2).sum()/denom) if denom else None,
            'wMAPE_pct': float(100*np.abs(y-pred).sum()/np.abs(y).sum())}


def summarize(frame):
    models = [c for c in frame if c.startswith('pred_')]
    rows, by_origin = [], []
    for (regime, h), group in frame.groupby(['regime', 'h']):
        # Models unavailable in an entire regime are explicitly omitted (V3/V4 at h12).
        for model in models:
            if group[model].isna().all():
                continue
            rows.append(dict(regime=regime, h=int(h), model=model[5:], n_pairs=len(group),
                             n_origins=group.origin.nunique(), **scores(group, model)))
            for origin, block in group.groupby('origin'):
                by_origin.append(dict(regime=regime, h=int(h), origin=str(origin.date()),
                                      model=model[5:], **scores(block, model)))
    return pd.DataFrame(rows), pd.DataFrame(by_origin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/contest_prophet.json')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--sample', type=int, help='Override sample size; 0 means all municipalities')
    parser.add_argument('--skip-prophet', action='store_true', help='Infrastructure check only; NOT a Prophet benchmark')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/contest_prophet')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg['prediction_floor_rub'] != 0 or args.workers < 1:
        parser.error('This protocol requires floor=0 and workers >=1')
    if not args.skip_prophet:
        try:
            import prophet  # noqa: F401
        except ImportError as e:
            raise SystemExit('Prophet unavailable. Install requirements-prophet.txt or use notebooks/contest_prophet.ipynb. No comparison was run.') from e
    p = load_panel()
    count = cfg['sample_size'] if args.sample is None else args.sample
    if not 0 <= count <= len(p.values):
        parser.error('sample must be between 0 and the panel size')
    ids = np.arange(len(p.values)) if count == 0 else np.sort(np.random.default_rng(cfg['seed']).choice(len(p.values), count, replace=False))
    nat, wk = national_monthly(), weekly_monthly()
    models = {**final_models(), **{k:v for k,v in v4_models().items() if k == 'v4_diversified'}}
    # Deliberately separate output folder prevents a dry run overwriting completed evidence.
    out = args.output / ('without_prophet' if args.skip_prophet else 'with_prophet')
    if (out/'manifest.json').exists():
        raise SystemExit(f'Output already exists: {out}. Choose a fresh --output path to preserve evidence.')
    out.mkdir(parents=True, exist_ok=True)
    selected = p.meta.iloc[ids].copy()
    selected.insert(0, 'series', ids)
    selected.to_csv(out/'selected_series.csv', index=False)
    manifest = {'status':'running', 'prophet_executed':not args.skip_prophet, 'config':cfg,
                'sample_size':len(ids), 'category':'Все категории',
                'protocol':'retrospective exploratory; not untouched test or official organizer score',
                'selection':'complete positive 24-month series; retrospective completeness selection',
                'exogenous':'current-vintage national data truncated by observation month, not historical release vintage',
                'data_sha256':hashlib.sha256(find('potrebitelskie-beznalicnye*.csv').read_bytes()).hexdigest(),
                'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__), ROOT/'src/sbx/models.py', ROOT/'src/sbx/candidates.py', ROOT/'src/sbx/final.py', ROOT/'src/sbx/data.py']},
                'raw_data_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'data/raw').glob('*.csv'))},
                'versions':{name:importlib.metadata.version(name) for name in ['numpy','pandas'] + ([] if args.skip_prophet else ['prophet','cmdstanpy'])}}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    frames = []
    for t, hs, regime in evaluation_origins(p.periods, cfg['horizons'], cfg['min_history']):
        origin = p.periods[t]
        hist = p.logs[:, :t+1].copy()
        hist.setflags(write=False)
        ctx = {'origin':origin, 'national':nat.loc[:origin], 'weekly':wk.loc[:origin]}
        predictions = {'seasonal_naive':{h:hist[:, t+h-12] for h in hs}}
        if regime == 'main':
            predictions.update({name:fn(hist, hs, ctx) for name, fn in models.items()})
        else:
            # Explicit cold-start alternative, NOT a result of V3/V4.
            spend = ctx['national'].nat_spend.dropna()
            yoy = np.log(spend.iloc[-1]/spend.loc[spend.index[-1]-pd.DateOffset(years=1)])
            predictions['national_yoy_fallback'] = {h:hist[:, t+h-12]+yoy for h in hs}
        block = pd.concat([pd.DataFrame({'series':ids, 'origin':origin, 'h':h,
            'target_date':p.periods[t+h], 'regime':regime, 'y':p.values[ids,t+h],
            **{'pred_'+name:np.maximum(0, np.exp(pred[h][ids])) for name,pred in predictions.items()}}) for h in hs], ignore_index=True)
        if not args.skip_prophet:
            for variant in cfg['prophet_variants']:
                tasks = [(int(i), list(p.periods[:t+1]), p.values[i,:t+1], hs, variant, cfg['seed']) for i in ids]
                if args.workers == 1:
                    fitted = list(map(fit_prophet, tasks))
                else:
                    with ProcessPoolExecutor(args.workers) as executor:
                        fitted = list(executor.map(fit_prophet, tasks, chunksize=8))
                mapping = {(i,h):float(v) for i,preds in fitted for h,v in zip(hs,preds)}
                block['pred_prophet_'+variant] = [mapping[i,h] for i,h in zip(block.series,block.h)]
        frames.append(block)
        # Per-origin checkpoint; no reuse without explicit provenance validation.
        block.to_csv(out/f'pairs_{origin:%Y%m}.csv.gz', index=False)
        print(f'{origin:%Y-%m}: {regime}, horizons={hs}, pairs={len(block)}', flush=True)
    frame = pd.concat(frames, ignore_index=True)
    metrics, per_origin = summarize(frame)
    metrics.to_csv(out/'metrics.csv', index=False)
    per_origin.to_csv(out/'metrics_by_origin.csv', index=False)
    comparisons = []
    for (regime,h), group in metrics.groupby(['regime','h']):
        for _, base in group[group.model.str.startswith('prophet_')].iterrows():
            for _, candidate in group[~group.model.str.startswith('prophet_')].iterrows():
                comparisons.append({'regime':regime, 'h':int(h), 'model':candidate.model, 'baseline':base.model,
                    'MAE_reduction_pct':100*(1-candidate.MAE/base.MAE) if base.MAE else None})
    pd.DataFrame(comparisons, columns=['regime','h','model','baseline','MAE_reduction_pct']).to_csv(out/'comparisons.csv', index=False)
    manifest['status'] = 'completed_without_prophet' if args.skip_prophet else 'completed'
    manifest['n_pairs'] = len(frame)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()
