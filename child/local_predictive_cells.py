"""Small local next-state predictors with persistent receiver traces.

This is deliberately not a transformer and not a biological neuron model.
Each output cell predicts only its own next activation from a fixed-radius
neighbourhood.  Every cell owns its parameters; no global attention matrix is
formed.  Learning is a one-step local delta rule, not BPTT/autograd.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RingWorld:
    n_cells: int = 24
    reversal_mean: float = 80.0
    seed: int = 0

    def generate(self, steps: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        pos = int(rng.integers(self.n_cells))
        direction = int(rng.choice((-1, 1)))
        x = np.zeros((steps, self.n_cells), dtype=np.float64)
        for t in range(steps):
            x[t, pos] = 1.0
            pos = (pos + direction) % self.n_cells
            if rng.random() < 1.0 / self.reversal_mean:
                direction *= -1
        return x


class LocalPredictiveCells:
    """A ring of individually parameterized local next-state predictors."""

    def __init__(
        self,
        n_cells: int = 24,
        radius: int = 2,
        trace_decay: float = 0.40,
        learning_rate: float = 0.03,
        use_trace: bool = True,
        shuffle_trace: bool = False,
        seed: int = 0,
    ) -> None:
        self.n_cells = n_cells
        self.radius = radius
        self.trace_decay = trace_decay
        self.learning_rate = learning_rate
        self.use_trace = use_trace
        self.shuffle_trace = shuffle_trace
        self.offsets = np.arange(-radius, radius + 1)
        centers = np.arange(n_cells)[:, None]
        self.neighbour_indices = (centers + self.offsets[None, :]) % n_cells
        self.trace = np.zeros(n_cells, dtype=np.float64)
        self.rng = np.random.default_rng(seed + 100_003)

        n_features = len(self.offsets) * (2 if use_trace else 1) + 1
        init_rng = np.random.default_rng(seed)
        self.weights = init_rng.normal(0.0, 1e-3, size=(n_cells, n_features))

    def reset_fast_state(self) -> None:
        self.trace.fill(0.0)

    def _features(self, x: np.ndarray) -> np.ndarray:
        local_x = x[self.neighbour_indices]
        pieces = [local_x]
        if self.use_trace:
            trace = self.trace
            if self.shuffle_trace:
                trace = trace[self.rng.permutation(self.n_cells)]
            pieces.append(trace[self.neighbour_indices])
        pieces.append(np.ones((self.n_cells, 1), dtype=np.float64))
        return np.concatenate(pieces, axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        features = self._features(x)
        return np.sum(self.weights * features, axis=1)

    def update_fast_state(self, x: np.ndarray) -> None:
        a = self.trace_decay
        self.trace = a * self.trace + (1.0 - a) * x

    def learn_one_step(self, x_now: np.ndarray, x_next: np.ndarray) -> np.ndarray:
        """Predict next local activations, then apply a postsynaptic local delta."""
        features = self._features(x_now)
        prediction = np.sum(self.weights * features, axis=1)
        local_error = x_next - prediction
        self.weights += self.learning_rate * local_error[:, None] * features
        self.update_fast_state(x_now)
        return prediction

    def autonomous_step(self, x_now: np.ndarray) -> np.ndarray:
        """Feed the locally predicted winner back as the next network state."""
        prediction = self.predict(x_now)
        next_state = np.zeros(self.n_cells, dtype=np.float64)
        next_state[int(np.argmax(prediction))] = 1.0
        self.update_fast_state(x_now)
        return next_state
