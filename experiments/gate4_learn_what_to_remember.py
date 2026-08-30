"""Gate 4: learn what deserves slow memory.

Every block presents 100 candidate experiences but slow memory can retain only
20.  At write time the future query/relevance outcome is unknown.

Each event has observable features whose combination statistically predicts
future relevance.  A small online relevance model learns from delayed query
outcomes and ranks events for consolidation.

This is a resource-allocation / caching experiment, not a model of human IQ.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


BLOCKS = 400
BLOCK_SIZE = 100
MEMORY_BUDGET = 20
FEATURE_DIM = 6
WARMUP_BLOCKS = 50
LEARNING_RATE = 0.08
SEEDS = 20


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def memory_features(x: np.ndarray) -> np.ndarray:
    """Small nonlinear basis available to the relevance learner."""
    return np.c_[
        x,
        x[:, 1] * x[:, 2],
        np.sin(x[:, 4]),
        x[:, 0] ** 2,
        np.ones(len(x)),
    ]


def run_seed(seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_phi = FEATURE_DIM + 4
    weights = np.zeros(n_phi, dtype=np.float64)

    retained_queries = {
        "random": 0,
        "salience": 0,
        "learned": 0,
        "oracle": 0,
        "full": 0,
    }
    total_queries = 0

    for block_id in range(BLOCKS):
        x = rng.normal(size=(BLOCK_SIZE, FEATURE_DIM))

        # Hidden law controlling whether this experience will be queried later.
        # The learner sees x but not this equation or the future Bernoulli draw.
        true_logit = (
            1.5 * x[:, 0]
            + 1.0 * x[:, 1] * x[:, 2]
            - 0.9 * x[:, 3]
            + 0.7 * np.sin(x[:, 4])
            - 1.2
        )
        future_query_probability = sigmoid(true_logit)
        future_query = (
            rng.random(BLOCK_SIZE) < future_query_probability
        )

        phi = memory_features(x)
        predicted_value = sigmoid(phi @ weights)

        selected = {
            "random": rng.choice(
                BLOCK_SIZE, MEMORY_BUDGET, replace=False
            ),
            # A plausible but incomplete current-salience heuristic.
            "salience": np.argsort(x[:, 0])[-MEMORY_BUDGET:],
            "learned": np.argsort(predicted_value)[-MEMORY_BUDGET:],
            # Positive control: knows the true probability, but not the
            # realized future query.
            "oracle": np.argsort(
                future_query_probability
            )[-MEMORY_BUDGET:],
            "full": np.arange(BLOCK_SIZE),
        }

        if block_id >= WARMUP_BLOCKS:
            total_queries += int(future_query.sum())
            for name, indices in selected.items():
                retained_queries[name] += int(
                    future_query[indices].sum()
                )

        # Only after the retention decision does delayed relevance become
        # observable.  Learn the future-value predictor for later blocks.
        error = future_query.astype(float) - predicted_value
        weights += (
            LEARNING_RATE
            * (phi.T @ error)
            / BLOCK_SIZE
        )

    return {
        name: retained / total_queries
        for name, retained in retained_queries.items()
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for name in rows[0]:
        values = np.array([row[name] for row in rows], dtype=float)
        out[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return out


def main() -> None:
    per_seed = [run_seed(seed) for seed in range(SEEDS)]
    summary = aggregate(per_seed)

    print("Gate 4 — learn what to remember")
    for name, stats in summary.items():
        print(
            f"{name:12s} future-query recall "
            f"{stats['mean']:.4f} ± {stats['std']:.4f}"
        )

    receipt = {
        "gate": 4,
        "blocks": BLOCKS,
        "block_size": BLOCK_SIZE,
        "memory_budget": MEMORY_BUDGET,
        "memory_fraction": MEMORY_BUDGET / BLOCK_SIZE,
        "warmup_blocks": WARMUP_BLOCKS,
        "learning_rate": LEARNING_RATE,
        "seeds": SEEDS,
        "summary": summary,
        "per_seed": per_seed,
    }

    out = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "gate4_learn_what_to_remember.json"
    )
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
