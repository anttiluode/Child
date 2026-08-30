"""Action-conditioned local predictor plus an AIS-like local output gate.

Gate 1 deliberately separates two roles:
- somatodendritic-like local state predicts what comes next;
- an output gate decides whether the current pulse is transmitted.

The gate is not claimed to be a biological AIS model.  It is a minimal control
boundary inspired by the computational separation between rich local state and
regulated axonal output.
"""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


class ActionConditionedPredictor:
    """Per-cell one-step predictor conditioned on local history and output action."""

    def __init__(
        self,
        n_cells: int = 24,
        radius: int = 2,
        trace_decay: float = 0.40,
        learning_rate: float = 0.03,
        seed: int = 0,
    ) -> None:
        self.n_cells = n_cells
        self.radius = radius
        self.trace_decay = trace_decay
        self.learning_rate = learning_rate
        offsets = np.arange(-radius, radius + 1)
        centers = np.arange(n_cells)[:, None]
        self.neighbour_indices = (centers + offsets[None, :]) % n_cells
        self.trace = np.zeros(n_cells, dtype=np.float64)

        # current local activity + local trace + transmit action + bias
        n_features = 2 * len(offsets) + 2
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 1e-3, size=(n_cells, n_features))

    def _features(self, x: np.ndarray, transmit: int) -> np.ndarray:
        local_x = x[self.neighbour_indices]
        local_trace = self.trace[self.neighbour_indices]
        action = np.full((self.n_cells, 1), float(transmit))
        bias = np.ones((self.n_cells, 1), dtype=np.float64)
        return np.concatenate((local_x, local_trace, action, bias), axis=1)

    def predict(self, x: np.ndarray, transmit: int) -> np.ndarray:
        features = self._features(x, transmit)
        return np.einsum("ij,ij->i", self.weights, features)

    def learn_one_step(
        self, x_now: np.ndarray, transmit: int, x_next: np.ndarray
    ) -> np.ndarray:
        features = self._features(x_now, transmit)
        prediction = np.einsum("ij,ij->i", self.weights, features)
        error = x_next - prediction
        self.weights += self.learning_rate * error[:, None] * features
        self.update_fast_state(x_now)
        return prediction

    def update_fast_state(self, x: np.ndarray) -> None:
        a = self.trace_decay
        self.trace = a * self.trace + (1.0 - a) * x


class LocalOutputGate:
    """Independent stochastic transmit/hold gate for every ring location.

    Prediction reward uses a REINFORCE-style local scalar update.  The optional
    homeostatic term directly pushes expected transmission toward a target.
    Neither update differentiates through the predictor or the world.
    """

    def __init__(
        self,
        n_cells: int = 24,
        learning_rate: float = 0.05,
        homeostasis_strength: float = 0.0,
        target_transmission: float = 0.80,
    ) -> None:
        self.logits = np.zeros(n_cells, dtype=np.float64)
        self.baseline = np.zeros(n_cells, dtype=np.float64)
        self.learning_rate = learning_rate
        self.homeostasis_strength = homeostasis_strength
        self.target_transmission = target_transmission

    def probability(self, cell: int) -> float:
        return float(sigmoid(self.logits[cell]))

    def choose(self, cell: int, rng: np.random.Generator) -> tuple[int, float]:
        p = self.probability(cell)
        return int(rng.random() < p), p

    def update(
        self,
        cell: int,
        transmit: int,
        p_transmit: float,
        prediction_reward: float,
    ) -> None:
        advantage = prediction_reward - self.baseline[cell]
        self.baseline[cell] = 0.98 * self.baseline[cell] + 0.02 * prediction_reward

        policy = self.learning_rate * advantage * (transmit - p_transmit)
        homeostasis = self.homeostasis_strength * (
            self.target_transmission - p_transmit
        )
        self.logits[cell] = np.clip(
            self.logits[cell] + policy + homeostasis, -6.0, 6.0
        )

    @property
    def mean_probability(self) -> float:
        return float(np.mean(sigmoid(self.logits)))
