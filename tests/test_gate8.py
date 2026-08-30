import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gate8",
    ROOT / "experiments" / "gate8_constitutional_scientist.py",
)
gate8 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gate8
SPEC.loader.exec_module(gate8)


class Gate8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.identifiable = [gate8.run_seed(seed) for seed in range(8)]
        cls.aliased = [gate8.run_seed(seed, True) for seed in range(4)]

    def test_incumbent_selected_queries_hide_every_pair_interaction(self):
        world = gate8.make_world(0)
        evidence, peak = gate8.collect_delayed_evidence(
            world, 0, "captured"
        )
        self.assertEqual(peak, gate8.AUDIT_DELAY)
        for row in evidence:
            nonzero = np.flatnonzero(row.x)
            self.assertEqual(len(nonzero), 1)
            for law in gate8.candidates():
                self.assertEqual(
                    gate8.interaction_value(law.pair, row.x), 0.0
                )

    def test_coverage_detects_misspecification_while_capture_does_not(self):
        for row in self.identifiable:
            self.assertEqual(
                row["captured_incumbent"]["mismatch_detected"], 0
            )
            self.assertGreater(
                row["captured_incumbent"]["noiseless_dense_rmse"], 0.79
            )
            self.assertEqual(
                row["fixed_catalog_coverage"]["mismatch_detected"], 1
            )
            self.assertEqual(
                row["fixed_catalog_coverage"]["status"], "MODEL_MISMATCH"
            )

    def test_bounded_trace_scores_newborn_law_without_new_probes(self):
        for row in self.identifiable:
            result = row["constitutional_trace"]
            self.assertEqual(result["status"], "IDENTIFIED")
            self.assertEqual(result["truth_covered"], 1)
            self.assertEqual(result["post_birth_probes"], 0)
            self.assertLess(result["noiseless_dense_rmse"], 1e-12)

        active_probes = [
            row["no_replay_active"]["post_birth_probes"]
            for row in self.identifiable
        ]
        self.assertGreater(float(np.mean(active_probes)), 3.0)

    def test_bounded_trace_beats_full_audit_ledger_memory(self):
        for row in self.identifiable:
            bounded = row["constitutional_trace"]["peak_raw_scalars"]
            full_audit = row["full_audit_ledger"]["peak_raw_scalars"]
            full_history = row["full_history"]["peak_raw_scalars"]
            self.assertGreaterEqual(full_audit / bounded, 4.0)
            self.assertGreater(full_history / bounded, 14.0)

    def test_birth_time_recovery_depends_on_trace_capacity(self):
        sweep = gate8.capacity_sweep(8)
        self.assertEqual(sweep["0"]["truth_recovery_fraction"], 0.0)
        self.assertEqual(sweep["16"]["truth_recovery_fraction"], 1.0)
        self.assertLess(
            sweep["4"]["truth_recovery_fraction"],
            sweep["16"]["truth_recovery_fraction"],
        )

    def test_destroying_causal_addresses_produces_wrong_certainty(self):
        accuracy = np.mean(
            [
                row["shuffled_address_trace"]["truth_covered"]
                for row in self.identifiable
            ]
        )
        confidence = np.mean(
            [
                row["shuffled_address_trace"]["top_class_mass"]
                for row in self.identifiable
            ]
        )
        self.assertLess(accuracy, 0.50)
        self.assertGreater(confidence, 0.95)

    def test_equivalent_laws_are_refused_not_arbitrarily_selected(self):
        for row in self.aliased:
            result = row["constitutional_trace"]
            self.assertEqual(result["status"], "NOT_IDENTIFIABLE")
            self.assertEqual(result["winner_class_size"], 2)
            self.assertEqual(result["truth_covered"], 1)
            self.assertLess(result["noiseless_dense_rmse"], 1e-12)


if __name__ == "__main__":
    unittest.main()
