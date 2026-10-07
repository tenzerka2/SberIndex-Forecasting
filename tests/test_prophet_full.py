import json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
from prophet_full_shard import shard_ids,verified_cache
from aggregate_prophet_full import validate_partition
from prepare_contest_bundle import load_bundle


class FullPanelTests(unittest.TestCase):
    def test_shards_cover_every_series_once(self):
        ids=np.concatenate([shard_ids(2016,s,32) for s in range(32)])
        np.testing.assert_array_equal(np.sort(ids),np.arange(2016))
        self.assertEqual(len(set(ids)),2016)
        with self.assertRaises(ValueError):shard_ids(2016,32,32)

    def test_missing_duplicate_and_incomplete_shards_fail(self):
        rows=[{'variant':'auto','shard':s,'shards':2,'status':'completed','prophet_executed':True,
               'panel_size':4,'sample_size':2,'n_pairs':48} for s in range(2)]
        validate_partition(rows,'auto',4,2)
        for bad in [rows[:1],[rows[0],rows[0]],[rows[0],{**rows[1],'status':'running'}],
                    [rows[0],{**rows[1],'n_pairs':47}]]:
            with self.assertRaises(ValueError):validate_partition(bad,'auto',4,2)

    def test_published_cache_matches_actual_inputs_and_rejects_version_drift(self):
        p,_,_,bundle=load_bundle(ROOT/'data/benchmark')
        cfg=json.loads((ROOT/'config/contest_prophet.json').read_text())
        versions=json.loads((ROOT/'outputs/prophet_verified/provenance.json').read_text())['manifests'][0]['versions']
        frame,info=verified_cache(ROOT/'outputs/prophet_verified',p,bundle,versions,cfg)
        self.assertEqual(frame.series.nunique(),100);self.assertEqual(len(frame),2400)
        self.assertTrue(info['run_url'])
        with self.assertRaises(ValueError):verified_cache(ROOT/'outputs/prophet_verified',p,bundle,{**versions,'prophet':'wrong'},cfg)
        changed=p.values.copy();p.values[frame.series.iloc[0],p.periods.get_loc(frame.target_date.iloc[0])]+=1
        with self.assertRaises(ValueError):verified_cache(ROOT/'outputs/prophet_verified',p,bundle,versions,cfg)
        p.values[:]=changed


if __name__=='__main__':unittest.main()
