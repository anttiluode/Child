import unittest
import numpy as np

from child.coupled_predictor import ActionConditionedPredictor, LocalOutputGate


class Gate1Tests(unittest.TestCase):
    def test_action_is_part_of_predictive_coordinates(self):
        m = ActionConditionedPredictor(n_cells=12, radius=2, seed=1)
        x = np.zeros(12)
        x[4] = 1.0
        f_hold = m._features(x, 0)
        f_send = m._features(x, 1)
        self.assertFalse(np.allclose(f_hold, f_send))

    def test_homeostasis_pushes_closed_gate_up(self):
        gate = LocalOutputGate(
            n_cells=4,
            homeostasis_strength=0.05,
            target_transmission=0.8,
        )
        gate.logits[:] = -3.0
        before = gate.mean_probability
        gate.update(0, transmit=0, p_transmit=gate.probability(0), prediction_reward=1.0)
        self.assertGreater(gate.probability(0), before)

    def test_prediction_only_has_no_homeostatic_push(self):
        gate = LocalOutputGate(n_cells=4, homeostasis_strength=0.0)
        self.assertEqual(gate.homeostasis_strength, 0.0)


if __name__ == "__main__":
    unittest.main()
