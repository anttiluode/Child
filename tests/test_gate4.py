import unittest

from experiments.gate4_learn_what_to_remember import run_seed


class Gate4Tests(unittest.TestCase):
    def test_learned_gate_beats_random(self):
        result = run_seed(0)
        self.assertGreater(result["learned"], result["random"] + 0.20)

    def test_learned_gate_beats_simple_salience(self):
        result = run_seed(0)
        self.assertGreater(result["learned"], result["salience"] + 0.05)

    def test_budgeted_oracle_beats_random(self):
        result = run_seed(0)
        self.assertGreater(result["oracle"], result["random"] + 0.20)


if __name__ == "__main__":
    unittest.main()
