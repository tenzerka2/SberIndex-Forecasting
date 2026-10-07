import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from contest_timesfm import context,project

class TestFoundationProtocol(unittest.TestCase):
    def test_future_cannot_change_context_or_yoy_projection(self):
        a=np.arange(1,49,dtype=float).reshape(2,24);b=a.copy();b[:,14:]=999999
        for variant in ['raw','yoy']:np.testing.assert_array_equal(context(a,13,variant),context(b,13,variant))
        point=np.zeros((2,12));q=np.zeros((2,12,10))
        out=project(a,13,[1,3,6],'yoy',point,q)
        other=project(b,13,[1,3,6],'yoy',point,q)
        for x,y in zip(out,other):np.testing.assert_array_equal(x,y)
        np.testing.assert_array_equal(out[0],a[:,[2,4,7]])
        with self.assertRaises(ValueError):context(a,11,'yoy')
    def test_quantile_indexes_and_nonnegative_constraint(self):
        a=np.ones((2,24));point=np.ones((2,12));q=np.zeros((2,12,10));q[:,:,1]=-2;q[:,:,9]=7
        pred,lo,hi=project(a,13,[1,6],'raw',point,q)
        np.testing.assert_array_equal(lo,np.zeros((2,2)));np.testing.assert_array_equal(hi,np.full((2,2),7))
        q[:,:,1]=8
        with self.assertRaises(ValueError):project(a,13,[1],'raw',point,q)
        point[0,0]=np.nan
        with self.assertRaises(ValueError):project(a,13,[1],'raw',point,q)

if __name__=='__main__':unittest.main()
