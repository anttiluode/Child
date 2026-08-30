import unittest
import numpy as np

from child.local_predictive_cells import LocalPredictiveCells, RingWorld


class Gate0Tests(unittest.TestCase):
    def test_world_is_one_hot(self):
        x = RingWorld(n_cells=12, seed=3).generate(100)
        np.testing.assert_allclose(x.sum(axis=1), 1.0)

    def test_local_parameter_count_is_linear_in_population(self):
        m = LocalPredictiveCells(n_cells=24, radius=2, use_trace=True)
        self.assertEqual(m.weights.shape, (24, 11))

    def test_trace_changes_prediction_coordinates(self):
        m = LocalPredictiveCells(n_cells=12, seed=1, use_trace=True)
        x = np.zeros(12)
        x[4] = 1.0
        before = m._features(x).copy()
        old = np.zeros(12)
        old[3] = 1.0
        m.update_fast_state(old)
        after = m._features(x)
        self.assertFalse(np.allclose(before, after))


if __name__ == "__main__":
    unittest.main()
