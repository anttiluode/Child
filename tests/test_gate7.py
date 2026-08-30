import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gate7",
    ROOT / "experiments" / "gate7_temporal_routing_body.py",
)
gate7 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gate7)


class Gate7Tests(unittest.TestCase):
    def test_all_bodies_have_exact_edge_budget(self):
        offsets = gate7.learn_lag_offsets()
        for body in gate7.build_bodies(0, offsets).values():
            gate7.validate_body(body)
            self.assertEqual(body.shape, (gate7.N_NODES, gate7.OUT_DEGREE))

    def test_multiscale_body_reaches_every_ring_address_locally(self):
        body = gate7.circulant_body([1, 2, 4, 8, 16, 32])
        outcomes = [
            gate7.greedy_route(body, 0, target)[0]
            for target in range(1, gate7.N_NODES)
        ]
        self.assertTrue(all(outcomes))

    def test_random_support_has_reach_but_not_local_navigability(self):
        body = gate7.random_sparse_body(np.random.default_rng(10_000))
        shortest = gate7.all_pairs_shortest_hops(body)
        oracle = np.mean(shortest[0, 1:] <= gate7.HOP_BUDGET)
        greedy = np.mean(
            [
                gate7.greedy_route(body, 0, target)[0]
                for target in range(1, gate7.N_NODES)
            ]
        )
        self.assertGreater(oracle, 0.99)
        self.assertLess(greedy, 0.25)

    def test_uniform_workload_separates_local_and_multiscale(self):
        result = gate7.run_seed(0, query_count=800)
        local = result["local"]["uniform"]["greedy_success"]
        dyadic = result["dyadic_multiscale"]["uniform"]["greedy_success"]
        learned = result["learned_lag_support"]["uniform"]["greedy_success"]
        random_local = result["random_sparse"]["uniform"]["greedy_success"]
        random_oracle = result["random_sparse"]["uniform"]["oracle_shortest_success"]

        self.assertLess(local, 0.50)
        self.assertGreater(dyadic, 0.99)
        self.assertGreater(learned, 0.99)
        self.assertLess(random_local, 0.30)
        self.assertGreater(random_oracle, 0.99)


if __name__ == "__main__":
    unittest.main()
