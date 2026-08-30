import unittest

from experiments.gate6_active_sensing_delayed_audit import run_seed


class Gate6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_seed(0)

    def test_oracle_observation_allocation_beats_always_sensing(self):
        self.assertGreater(
            self.result["oracle"]["post_late_utility"],
            self.result["always_sense"]["post_late_utility"] + 0.02,
        )

    def test_delayed_trace_learns_shifted_sensing_policy(self):
        learned = self.result["learned_trace"]
        self.assertGreater(learned["post_late_sense_context_0"], 0.80)
        self.assertLess(learned["post_late_sense_context_1"], 0.20)
        self.assertEqual(learned["recovered"], 1)

    def test_trace_approaches_zero_delay_control(self):
        delayed = self.result["learned_trace"]["post_late_utility"]
        immediate = self.result["zero_delay_learner"]["post_late_utility"]
        self.assertAlmostEqual(delayed, immediate, delta=0.015)

    def test_no_trace_remains_at_conservative_probe_policy(self):
        forgetful = self.result["learned_no_trace"]
        delayed = self.result["learned_trace"]
        self.assertGreater(forgetful["post_late_sense_rate"], 0.95)
        self.assertGreater(
            delayed["post_late_utility"],
            forgetful["post_late_utility"] + 0.02,
        )

    def test_delay_has_explicit_trace_cost(self):
        delayed = self.result["learned_trace"]
        immediate = self.result["zero_delay_learner"]
        self.assertEqual(delayed["peak_trace_records"], 12)
        self.assertEqual(delayed["peak_trace_scalars"], 60)
        self.assertEqual(immediate["peak_trace_records"], 0)


if __name__ == "__main__":
    unittest.main()
