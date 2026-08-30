"""Gate 0: local next-state prediction turns history into routing.

The observed world is only a one-hot pulse position on a ring.  Long runs move
clockwise or counter-clockwise and occasionally reverse.  Current position alone
is therefore insufficient to know which neighbour comes next.  Recent trajectory
history is sufficient.

The candidate is intentionally humble: every cell predicts only its own next
activation from a radius-2 local neighbourhood plus an exponentially decaying
receiver trace.  Each cell updates only from its own one-step prediction error.
There is no autograd, BPTT, global attention, or hidden-layer error transport.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from child.local_predictive_cells import LocalPredictiveCells, RingWorld


N_CELLS = 24
STEPS = 12_000
TRAIN_STEPS = 8_000
SEEDS = 8
REVERSAL_MEAN = 80.0


def frozen_accuracy(seed: int, mode: str) -> float:
    stream = RingWorld(N_CELLS, REVERSAL_MEAN, seed).generate(STEPS)
    kwargs = dict(n_cells=N_CELLS, seed=seed)
    if mode == "stateful":
        model = LocalPredictiveCells(**kwargs, use_trace=True)
    elif mode == "current_only":
        model = LocalPredictiveCells(**kwargs, use_trace=False)
    elif mode == "shuffled_trace":
        model = LocalPredictiveCells(**kwargs, use_trace=True, shuffle_trace=True)
    else:
        raise ValueError(mode)

    for t in range(TRAIN_STEPS - 1):
        model.learn_one_step(stream[t], stream[t + 1])

    correct = 0
    total = 0
    for t in range(TRAIN_STEPS, STEPS - 1):
        prediction = model.predict(stream[t])
        correct += int(np.argmax(prediction) == np.argmax(stream[t + 1]))
        total += 1
        model.update_fast_state(stream[t])
    return correct / total


def markov_attacker(seed: int) -> float:
    """Explicit previous-position rule: continue the observed direction."""
    stream = RingWorld(N_CELLS, REVERSAL_MEAN, seed).generate(STEPS)
    correct = 0
    total = 0
    prev = int(np.argmax(stream[TRAIN_STEPS - 1]))
    for t in range(TRAIN_STEPS, STEPS - 1):
        cur = int(np.argmax(stream[t]))
        delta = (cur - prev) % N_CELLS
        direction = 1 if delta == 1 else -1
        guess = (cur + direction) % N_CELLS
        truth = int(np.argmax(stream[t + 1]))
        correct += int(guess == truth)
        total += 1
        prev = cur
    return correct / total


def autonomous_rollout(seed: int, direction: int, rollout_steps: int = 96) -> float:
    """After training, two observed pulses seed a self-generated travelling wave."""
    stream = RingWorld(N_CELLS, REVERSAL_MEAN, seed).generate(12_000)
    model = LocalPredictiveCells(n_cells=N_CELLS, seed=seed, use_trace=True)
    for t in range(len(stream) - 1):
        model.learn_one_step(stream[t], stream[t + 1])

    model.reset_fast_state()
    start = 5
    first = np.zeros(N_CELLS)
    second = np.zeros(N_CELLS)
    first[start] = 1.0
    second[(start + direction) % N_CELLS] = 1.0
    model.update_fast_state(first)
    current = second
    expected_pos = (start + direction) % N_CELLS

    correct = 0
    for _ in range(rollout_steps):
        current = model.autonomous_step(current)
        expected_pos = (expected_pos + direction) % N_CELLS
        correct += int(np.argmax(current) == expected_pos)
    return correct / rollout_steps


def summarize(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0))}


def main() -> None:
    modes = ("current_only", "shuffled_trace", "stateful")
    per_seed = {mode: [frozen_accuracy(seed, mode) for seed in range(SEEDS)] for mode in modes}
    per_seed["explicit_markov"] = [markov_attacker(seed) for seed in range(SEEDS)]
    rollout = [
        autonomous_rollout(seed, direction)
        for seed in range(3)
        for direction in (-1, 1)
    ]

    receipt = {
        "gate": 0,
        "n_cells": N_CELLS,
        "steps": STEPS,
        "train_steps": TRAIN_STEPS,
        "reversal_mean": REVERSAL_MEAN,
        "summary": {name: summarize(values) for name, values in per_seed.items()},
        "autonomous_rollout": summarize(rollout),
        "per_seed": per_seed,
    }

    print("Gate 0 — local next-state prediction")
    for name in ("current_only", "shuffled_trace", "stateful", "explicit_markov"):
        s = receipt["summary"][name]
        print(f"{name:20s} {s['mean']:.4f} ± {s['std']:.4f}")
    s = receipt["autonomous_rollout"]
    print(f"{'autonomous_rollout':20s} {s['mean']:.4f} ± {s['std']:.4f}")

    out = ROOT / "results" / "gate0_local_next_state.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
