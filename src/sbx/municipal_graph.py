"""Official-ID alignment and static transport-neighbor diagnostics.

Full-history value matching is used only to recover the identity of an old ID-less
export. It is not a model feature or a method of predicting new municipalities.
The December 2024 graph cannot establish real-time availability at earlier origins.
"""
import numpy as np
import pandas as pd


def match_official_ids(panel, consumption):
    required={'territory_id','date','category','value'}
    if not required.issubset(consumption):raise ValueError('Missing official consumption columns')
    d=consumption[consumption.category.eq('Все категории')].copy()
    d['date']=pd.to_datetime(d.date)
    if d.duplicated(['territory_id','date']).any():raise ValueError('Duplicate territory-month')
    wide=d.pivot(index='territory_id',columns='date',values='value').reindex(columns=panel.periods)
    wide=wide.loc[wide.notna().all(axis=1)]
    if wide.duplicated().any():raise ValueError('Ambiguous official value signatures')
    lookup={tuple(row):int(i) for i,row in wide.iterrows()}
    ids=[lookup.get(tuple(row)) for row in panel.values]
    if any(i is None for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('Official rows do not uniquely match every legacy row')
    out=panel.meta.copy();out.insert(0,'series',np.arange(len(ids)))
    out['territory_id']=ids
    np.testing.assert_array_equal(wide.loc[ids].to_numpy(),panel.values)
    return out


def transport_neighbors(ids, connections, k=8):
    ids=np.asarray(ids);n=len(ids)
    if len(set(ids))!=n:raise ValueError('Non-unique territory IDs')
    if k<1 or k>=n:raise ValueError('Invalid neighbor count')
    lookup=pd.Series(np.arange(n),index=ids)
    d=connections.loc[connections.type.eq('highway'),['territory_id_x','territory_id_y','distance']]
    x=d.territory_id_x.map(lookup);y=d.territory_id_y.map(lookup)
    valid=x.notna() & y.notna() & x.ne(y) & np.isfinite(d.distance) & d.distance.gt(0)
    x=x[valid].to_numpy(int);y=y[valid].to_numpy(int);dist=d.loc[valid,'distance'].to_numpy(float)
    distance=np.full((n,n),np.inf)
    # The source stores one direction only. Retain the minimum if duplicate records exist.
    np.minimum.at(distance,(x,y),dist);np.minimum.at(distance,(y,x),dist)
    neighbors=np.argsort(distance,axis=1,kind='stable')[:,:k]
    km=np.take_along_axis(distance,neighbors,axis=1)
    weights=np.where(np.isfinite(km),1/np.maximum(km,1),0)
    weights/=np.maximum(weights.sum(axis=1,keepdims=True),1e-12)
    return neighbors,weights,km


def aggregate_neighbors(values, neighbors, weights):
    values=np.asarray(values)
    return np.sum(values[neighbors]*weights[:,:,None],axis=1)


def graph_features(base, neighbors, weights, names=('jump','slope','past_event')):
    f={f'neighbor_{name}':aggregate_neighbors(base[name],neighbors,weights) for name in names}
    T=next(iter(base.values())).shape[1]
    f['graph_available']=np.repeat((weights.sum(axis=1)>0)[:,None],T,axis=1).astype(float)
    return f


def permuted_neighbors(neighbors, weights, seed=47):
    """Relabel nodes jointly, preserving graph topology, degrees and absence of self edges."""
    p=np.random.default_rng(seed).permutation(len(neighbors));inv=np.argsort(p)
    return inv[neighbors[p]],weights[p]


def territory_metadata(mapping, dictionary, year):
    """Join the official version valid in [year_from, year_to), refusing ambiguity."""
    d=dictionary.loc[(dictionary.year_from<=year)&(dictionary.year_to>year)].copy()
    if d.territory_id.duplicated().any():raise ValueError('Overlapping dictionary versions')
    joined=mapping.merge(d,on='territory_id',how='left',validate='one_to_one',indicator=True)
    if not joined['_merge'].eq('both').all():raise ValueError('Dictionary does not cover all panel territories')
    return joined.drop(columns='_merge')
