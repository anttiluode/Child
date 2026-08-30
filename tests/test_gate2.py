import unittest

from experiments.gate2_delayed_relevance import run_seed


class Gate2Tests(unittest.TestCase):
    def test_hybrid_retains_delayed_relevance(self):
        result = run_seed(0)
        self.assertEqual(result["hybrid"]["first_accuracy"], 1.0)
        self.assertEqual(result["hybrid"]["repeat_accuracy"], 1.0)

    def test_fast_only_forgets_repeat(self):
        result = run_seed(0)
        self.assertEqual(result["fast_only"]["first_accuracy"], 1.0)
        self.assertEqual(result["fast_only"]["repeat_accuracy"], 0.0)

    def test_hybrid_uses_less_peak_storage_than_history(self):
        result = run_seed(0)
        self.assertLess(
            result["hybrid"]["peak_stored_scalars"],
            result["full_history"]["peak_stored_scalars"],
        )


if __name__ == "__main__":
    unittest.main()
