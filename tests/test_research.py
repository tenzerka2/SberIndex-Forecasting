"""Meaningful regression checks requiring only NumPy/pandas and the raw export."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx import early_warning as EW, backtest as B
from sbx.data import load_panel, national_monthly
from sbx.candidates import research_models, v4_models, national_shape
from sbx.final import final_models
from sbx.research_warning import average_precision, logistic_predict, top_budget
from sbx.intervals import conformal


class SyntheticTests(unittest.TestCase):
    def test_event_matures_and_never_revises(self):
        L = np.zeros((5, 30)); L[0, 8:] = .3; L[0, 20:] = .6
        full = EW.events(L)
        self.assertTrue(full[0, 8])
        self.assertFalse(EW.events(L[:, :10])[0, 8])
        self.assertTrue(EW.events(L[:, :11])[0, 8])
        for t in range(10, 30):
            np.testing.assert_array_equal(EW.events(L[:, :t + 1])[:, :t - 1], full[:, :t - 1])

    def test_unknown_labels_not_negative(self):
        L = np.zeros((5, 16))
        for task, delay in [("detect", 2), ("predict", 5)]:
            y=EW.mature_labels(L,task)
            self.assertTrue((y[:, -delay:] == -1).all())
            self.assertTrue((y[:, 6:-delay] == 0).all())

    def test_tied_average_precision(self):
        self.assertAlmostEqual(average_precision([1,0,1,0],[.5,.5,.5,.5]), .5)
        self.assertAlmostEqual(average_precision([1,0,1],[.9,.8,.7]), (1+2/3)/2)

    def test_alarm_budget_exact_and_deterministic(self):
        a=top_budget(np.ones(2016),.02)
        self.assertEqual(a.sum(),40)
        np.testing.assert_array_equal(a,top_budget(np.ones(2016),.02))

    def test_logistic_learns_direction(self):
        x=np.linspace(-3,3,200)[:,None]; y=(x[:,0]>0).astype(float)
        p=logistic_predict(x,y,np.array([[-2],[2]]))
        self.assertLess(p[0],.2); self.assertGreater(p[1],.8)


class RawDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel=load_panel(); cls.nat=national_monthly()

    def test_original_v3_reproduces(self):
        d=B.run(self.panel,{"v3":final_models()["v3"]})
        self.assertAlmostEqual(B.metrics(d,"v3")["MAE"],719.870192,places=4)

    def test_all_candidates_ignore_future(self):
        p=self.panel; t=17; o=p.periods[t]
        H=p.logs[:,:t+1]
        bad=self.nat.copy();bad.loc[bad.index>o]=1e9
        models={**research_models(),**v4_models(),"national_shape":national_shape}
        for name,fn in models.items():
            a=fn(H,[1,3,6],{"origin":o,"national":self.nat.loc[:o]})
            b=fn(H,[1,3,6],{"origin":o,"national":bad})
            for h in a:
                np.testing.assert_array_equal(a[h],b[h],err_msg=name)
                self.assertTrue(np.isfinite(a[h]).all())

    def test_actual_mature_labels_stable(self):
        L=self.panel.logs
        for task in ["detect","predict"]:
            full=EW.mature_labels(L,task)
            for t in range(12,23):
                part=EW.mature_labels(L[:,:t+1],task)
                known=part>=0
                np.testing.assert_array_equal(part[known],full[:,:t+1][known])

    def test_warning_features_ignore_future(self):
        from sbx.research_warning import features
        L=self.panel.logs[:100]; bad=L.copy();bad[:,18:]=100
        a,b=features(L),features(bad)
        for k in a:
            np.testing.assert_array_equal(a[k][:,:18],b[k][:,:18],err_msg=k)

    def test_local_news_ignore_future(self):
        from sbx.news_local import tensor
        p=self.panel;t=17
        a,_,_=tensor(p);b,_,_=tensor(p,cutoff=p.periods[t])
        for k in a:
            np.testing.assert_array_equal(a[k][:,:t+1],b[k][:,:t+1],err_msg=k)

    def test_empirical_intervals_ignore_future_errors(self):
        p=self.panel; o=pd.Timestamp("2024-07-01")
        d=B.run(p,{"v3":final_models()["v3"]},B.EXTENDED_ORIGINS)
        bad=d.copy();bad.loc[bad.target_date>o,"y"]*=3
        a=conformal(d,"v3",p.logs,p.periods);b=conformal(bad,"v3",p.logs,p.periods)
        for c in ["v3_lo80","v3_hi80","v3_lo90","v3_hi90"]:
            np.testing.assert_allclose(a.loc[a.origin.eq(o),c],b.loc[b.origin.eq(o),c])


if __name__=="__main__": unittest.main(verbosity=2)
