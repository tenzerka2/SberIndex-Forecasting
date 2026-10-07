"""Pack only the exact inputs needed for reproducible CI, with source hashes."""
from pathlib import Path
import sys, json, gzip, base64, hashlib, io
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sbx.data import Panel, load_panel, national_monthly, weekly_monthly


def load_bundle(directory):
    directory = Path(directory)
    encoded = ''.join(p.read_text() for p in sorted(directory.glob('part_*.b64')))
    payload = gzip.decompress(base64.b64decode(encoded, validate=True))
    manifest = json.loads((directory/'manifest.json').read_text())
    if hashlib.sha256(payload).hexdigest() != manifest['payload_sha256']:
        raise ValueError('Prepared data hash mismatch')
    data = json.loads(payload)
    p = Panel(np.asarray(data['values'],dtype=float), pd.DatetimeIndex(data['periods']), pd.DataFrame(data['meta']))
    if p.values.shape != (2016,24) or not np.isfinite(p.values).all() or not (p.values>0).all():
        raise ValueError('Invalid municipal panel')
    frames = [pd.read_json(io.StringIO(data[k]),orient='split') for k in ['national','weekly']]
    for frame in frames:
        frame.index = pd.to_datetime(frame.index)
    return p, *frames, manifest


def main():
    p=load_panel(); nat=national_monthly(); wk=weekly_monthly()
    data={'values':p.values.tolist(),'periods':p.periods.strftime('%Y-%m-%d').tolist(),
          'meta':p.meta.to_dict(orient='records'),
          'national':nat.to_json(orient='split',date_format='iso',double_precision=15),
          'weekly':wk.to_json(orient='split',date_format='iso',double_precision=15)}
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    encoded=base64.b64encode(gzip.compress(payload,mtime=0)).decode()
    out=ROOT/'data/benchmark';out.mkdir(exist_ok=True)
    for i,start in enumerate(range(0,len(encoded),48000)):
        (out/f'part_{i:03}.b64').write_text(encoded[start:start+48000])
    manifest={'payload_sha256':hashlib.sha256(payload).hexdigest(),
        'raw_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'data/raw').glob('*.csv'))},
        'description':'2016 complete positive municipal total-category rows, Jan2023-Dec2024; national and weekly panels. No interpolation or invented observations.',
        'selection':'same retrospective complete-series selection as load_panel', 'format':'gzip JSON encoded base64, ordered parts'}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    q,n,w,_=load_bundle(out)
    np.testing.assert_array_equal(q.values,p.values)
    pd.testing.assert_frame_equal(q.meta,p.meta)
    np.testing.assert_allclose(n.to_numpy(),nat.to_numpy(),rtol=1e-14,atol=1e-14,equal_nan=True)
    np.testing.assert_allclose(w.to_numpy(),wk.to_numpy(),rtol=1e-14,atol=1e-14,equal_nan=True)
    print(f'Verified exact municipal panel and external signals; {len(encoded)} bytes, {(len(encoded)+47999)//48000} parts')

if __name__=='__main__': main()
