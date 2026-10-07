import sys,unittest
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'benchmarks'))
from forecast_operational import predict_by_horizon
from prepare_contest_bundle import load_bundle
from sbx.final import final_models

class TestOperationalHorizons(unittest.TestCase):
    def test_main_unchanged_year_matches_actual_backtest(self):
        p,nat,wk,_=load_bundle(ROOT/'data/benchmark');t=23;origin=p.periods[t]
        ctx={'origin':origin,'national':nat.loc[:origin],'weekly':wk.loc[:origin]}
        pred,models=predict_by_horizon(p.logs,ctx)
        old=final_models()['v3_hedge'](p.logs,[1,3,6,12],ctx)
        for h in [1,3,6]:
            np.testing.assert_array_equal(pred[h],old[h]);self.assertEqual(models[h],'v3_hedge')
        self.assertEqual(models[12],'national_yoy_fallback')
        # Reproduce the only actually observed h12 evaluation, December 2023 -> December 2024.
        origin=p.periods[11];ctx={'origin':origin,'national':nat.loc[:origin],'weekly':wk.loc[:origin]}
        pred,models=predict_by_horizon(p.logs[:,:12],ctx)
        mae=np.abs(p.values[:,23]-np.exp(pred[12])).mean()
        m=pd.read_csv(ROOT/'outputs/prophet_full_summary/metrics.csv')
        expected=m[(m.model=='national_yoy_fallback')&(m.h==12)].MAE.iloc[0]
        np.testing.assert_allclose(mae,expected,rtol=1e-12)

if __name__=='__main__':unittest.main()
