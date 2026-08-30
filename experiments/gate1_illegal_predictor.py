"""Gate 1: prediction can make the coupled world easier instead of modeling it.

A travelling pulse has an output action at its current cell:

    TRANSMIT -> let the external ring dynamics advance
    HOLD     -> keep the pulse where it is

The one-step predictor is conditioned on that action.  Holding is therefore
easy to predict.  Each source cell has a stochastic local output gate trained
only by one-step prediction correctness.

If prediction is the only objective, the gate can improve its score by closing
and making the world nearly static.

A second arm adds a local homeostatic pressure toward continued transmission.
This is not claimed as an optimal controller; fixed gates are included as
boring attackers.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from child.coupled_predictor import ActionConditionedPredictor, LocalOutputGate


N_CELLS = 24
REVERSAL_P = 0.15
PRETRAIN_STEPS = 4_000
ADAPT_STEPS = 12_000
EVAL_STEPS = 4_000
SEEDS = 10

GATE_LR = 0.05
HOMEOSTASIS_STRENGTH = 0.05
TARGET_TRANSMISSION = 0.80


def one_hot(position: int) -> np.ndarray:
    x = np.zeros(N_CELLS, dtype=np.float64)
    x[position] = 1.0
    return x


def run_arm(seed: int, arm: str) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    predictor = ActionConditionedPredictor(n_cells=N_CELLS, seed=seed)

    position = int(rng.integers(N_CELLS))
    direction = int(rng.choice((-1, 1)))
    x = one_hot(position)

    # Give the predictor experience with both actions before coupling the
    # action policy to prediction reward.  This isolates the objective trap
    # from simple ignorance of the HOLD transition.
    for _ in range(PRETRAIN_STEPS):
        transmit = int(rng.random() < 0.5)
        if rng.random() < REVERSAL_P:
            direction *= -1
        next_position = (
            (position + direction) % N_CELLS if transmit else position
        )
        y = one_hot(next_position)
        predictor.learn_one_step(x, transmit, y)
        position, x = next_position, y

    homeostasis = HOMEOSTASIS_STRENGTH if arm == "prediction_homeostasis" else 0.0
    gate = LocalOutputGate(
        n_cells=N_CELLS,
        learning_rate=GATE_LR,
        homeostasis_strength=homeostasis,
        target_transmission=TARGET_TRANSMISSION,
    )

    def choose_action() -> tuple[int, float]:
        if arm == "forced_open":
            return 1, 1.0
        if arm == "fixed_80":
            return int(rng.random() < 0.80), 0.80
        return gate.choose(position, rng)

    # Coupled adaptation.
    for _ in range(ADAPT_STEPS):
        current_position = position
        transmit, p_transmit = choose_action()

        if rng.random() < REVERSAL_P:
            direction *= -1
        next_position = (
            (position + direction) % N_CELLS if transmit else position
        )
        y = one_hot(next_position)

        prediction = predictor.predict(x, transmit)
        prediction_reward = float(int(np.argmax(prediction)) == next_position)

        if arm in ("prediction_only", "prediction_homeostasis"):
            gate.update(
                current_position,
                transmit,
                p_transmit,
                prediction_reward,
            )

        predictor.learn_one_step(x, transmit, y)
        position, x = next_position, y

    # Freeze learned predictor/gate parameters.  Fast traces still evolve.
    movement = []
    accuracy = []

    for _ in range(EVAL_STEPS):
        transmit, _ = choose_action()

        if rng.random() < REVERSAL_P:
            direction *= -1
        next_position = (
            (position + direction) % N_CELLS if transmit else position
        )
        y = one_hot(next_position)

        prediction = predictor.predict(x, transmit)
        accuracy.append(float(int(np.argmax(prediction)) == next_position))
        movement.append(float(transmit))

        predictor.update_fast_state(x)
        position, x = next_position, y

    if arm == "forced_open":
        mean_gate = 1.0
    elif arm == "fixed_80":
        mean_gate = 0.80
    else:
        mean_gate = gate.mean_probability

    return {
        "movement_fraction": float(np.mean(movement)),
        "prediction_accuracy": float(np.mean(accuracy)),
        "mean_transmission_probability": float(mean_gate),
    }


def summarize(values: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = values[0].keys()
    return {
        key: {
            "mean": float(np.mean([v[key] for v in values])),
            "std": float(np.std([v[key] for v in values])),
        }
        for key in keys
    }


def main() -> None:
    arms = (
        "prediction_only",
        "prediction_homeostasis",
        "fixed_80",
        "forced_open",
    )
    per_seed = {
        arm: [run_arm(seed, arm) for seed in range(SEEDS)]
        for arm in arms
    }
    summary = {arm: summarize(values) for arm, values in per_seed.items()}

    receipt = {
        "gate": 1,
        "n_cells": N_CELLS,
        "reversal_probability": REVERSAL_P,
        "pretrain_steps": PRETRAIN_STEPS,
        "adapt_steps": ADAPT_STEPS,
        "eval_steps": EVAL_STEPS,
        "seeds": SEEDS,
        "gate_learning_rate": GATE_LR,
        "homeostasis_strength": HOMEOSTASIS_STRENGTH,
        "target_transmission": TARGET_TRANSMISSION,
        "summary": summary,
        "per_seed": per_seed,
    }

    print("Gate 1 — the illegal predictor")
    for arm in arms:
        m = summary[arm]["movement_fraction"]
        a = summary[arm]["prediction_accuracy"]
        print(
            f"{arm:24s} move {m['mean']:.4f} ± {m['std']:.4f}  "
            f"predict {a['mean']:.4f} ± {a['std']:.4f}"
        )

    out = ROOT / "results" / "gate1_illegal_predictor.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
