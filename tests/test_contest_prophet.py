import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('contest', Path(__file__).parents[1]/'benchmarks/contest_prophet.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ProtocolTests(unittest.TestCase):
    def test_observed_targets_and_history(self):
        periods = pd.date_range('2023-01-01', periods=24, freq='MS')
        origins = list(m.evaluation_origins(periods, [1,3,6,12]))
        counts = {h:0 for h in [1,3,6,12]}
        for t, hs, regime in origins:
            self.assertGreaterEqual(t+1, 14 if regime == 'main' else 12)
            for h in hs:
                self.assertLess(t+h, len(periods))
                self.assertGreaterEqual(t+h-12, 0)
                self.assertLessEqual(t+h-12, t)
                counts[h] += 1
        self.assertEqual(counts, {1:10,3:8,6:5,12:1})

    def test_partial_missing_predictions_fail(self):
        frame = pd.DataFrame({'y':[10.,20.], 'pred_model':[11.,np.nan]})
        with self.assertRaises(ValueError):
            m.scores(frame, 'pred_model')

    def test_scores_known_values(self):
        metrics = m.scores(pd.DataFrame({'y':[10.,20.], 'pred_model':[12.,18.]}), 'pred_model')
        self.assertEqual(metrics['MAE'], 2.)
        self.assertAlmostEqual(metrics['R2_level'], .84)

    def test_future_changes_do_not_change_frozen_model(self):
        from sbx.final import final_models
        rng = np.random.default_rng(7)
        history = np.log(1000) + rng.normal(0,.1,(20,24))
        changed = history.copy()
        changed[:,18:] += 10
        # V3 has no exogenous dependency; only the observed prefix is passed.
        fn = final_models()['v3']
        a = fn(history[:,:18], [1,3,6], None)
        b = fn(changed[:,:18], [1,3,6], None)
        for h in a:
            np.testing.assert_array_equal(a[h], b[h])

if __name__ == '__main__':
    unittest.main()
