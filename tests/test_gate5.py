import unittest

from experiments.gate5_policy_blindness_under_drift import run_seed


class Gate5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_seed(0)
        cls.hidden = run_seed(0, "strictly_hidden")

    def test_greedy_policy_does_not_discover_hidden_shift(self):
        self.assertEqual(self.result["greedy_selected"]["recovered"], 0)
        self.assertLess(
            self.result["greedy_selected"]["post_late_new_group_share"],
            0.10,
        )

    def test_surprise_burst_recovers_without_extra_memory(self):
        adaptive = self.result["surprise_burst"]
        greedy = self.result["greedy_selected"]
        self.assertEqual(adaptive["recovered"], 1)
        self.assertGreater(
            adaptive["post_late_recall"],
            greedy["post_late_recall"] + 0.25,
        )
        self.assertEqual(adaptive["persistent_slots"], 20)
        self.assertEqual(adaptive["extra_temporary_trace_slots"], 0)

    def test_fast_trace_recovers_quickly_but_pays_temporary_storage(self):
        fast = self.result["fast_trace"]
        self.assertEqual(fast["recovered"], 1)
        self.assertLessEqual(fast["recovery_blocks"], 10)
        self.assertEqual(fast["extra_temporary_trace_slots"], 80)

    def test_triggered_exploration_is_cheaper_than_fixed_reserve(self):
        self.assertLess(
            self.result["surprise_burst"]["exploration_writes_per_block"],
            self.result["fixed_reserve"]["exploration_writes_per_block"],
        )

    def test_surprise_cannot_detect_strictly_off_policy_improvement(self):
        self.assertEqual(self.hidden["surprise_burst"]["recovered"], 0)
        self.assertEqual(self.hidden["greedy_selected"]["recovered"], 0)
        self.assertEqual(self.hidden["fixed_reserve"]["recovered"], 1)
        self.assertEqual(self.hidden["fast_trace"]["recovered"], 1)


if __name__ == "__main__":
    unittest.main()
