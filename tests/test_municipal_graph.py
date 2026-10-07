import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sbx.data import Panel
from sbx.municipal_graph import match_official_ids,transport_neighbors,aggregate_neighbors,permuted_neighbors,territory_metadata


class MunicipalGraphTests(unittest.TestCase):
    def test_reordered_rows_match_by_values_not_position(self):
        p=Panel(np.array([[10.,20.],[30.,40.]]),pd.date_range('2023-01-01',periods=2,freq='MS'),pd.DataFrame({'run_id':[7,9],'mo':['same','same'],'homonym':[True,True]}))
        d=pd.DataFrame({'territory_id':[88,99,88,99],'date':['2023-02','2023-01','2023-01','2023-02'],'category':['Все категории']*4,'value':[40,10,30,20]})
        self.assertEqual(match_official_ids(p,d).territory_id.tolist(),[99,88])
        with self.assertRaises(ValueError):match_official_ids(p,pd.concat([d,d.iloc[:1]]))

    def test_one_direction_is_symmetrized_and_island_has_no_signal(self):
        d=pd.DataFrame({'territory_id_x':[20,30],'territory_id_y':[10,20],'distance':[2.,5.],'type':['highway']*2})
        nn,w,km=transport_neighbors([10,20,30,40],d,k=1)
        self.assertEqual(nn[:3,0].tolist(),[1,0,1])
        np.testing.assert_array_equal(w[:,0],[1,1,1,0])
        v=np.arange(12).reshape(4,3)
        a=aggregate_neighbors(v,nn,w)
        np.testing.assert_array_equal(a[0],v[1]);np.testing.assert_array_equal(a[3],np.zeros(3))
        b=v.copy();b[:,2]=999
        np.testing.assert_array_equal(aggregate_neighbors(b,nn,w)[:,:2],a[:,:2])

    def test_permutation_preserves_no_self_connections(self):
        nn=np.array([[1],[2],[3],[0]]);w=np.ones((4,1))
        pn,pw=permuted_neighbors(nn,w)
        self.assertTrue((pn[:,0]!=np.arange(4)).all())
        np.testing.assert_array_equal(pw.sum(axis=1),np.ones(4))

    def test_dictionary_interval_end_is_exclusive(self):
        m=pd.DataFrame({'territory_id':[1]})
        d=pd.DataFrame({'territory_id':[1,1],'year_from':[2023,2024],'year_to':[2024,9999],'name':['old','new']})
        self.assertEqual(territory_metadata(m,d,2024)['name'].tolist(),['new'])
        with self.assertRaises(ValueError):territory_metadata(m,pd.concat([d,d.iloc[[1]]]),2024)


if __name__=='__main__':unittest.main()
