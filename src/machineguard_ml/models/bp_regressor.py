from __future__ import annotations

import numpy as np


class BPRegressor:
    """A compact fully connected regressor trained with backpropagation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 16,
        learning_rate: float = 0.05,
        epochs: int = 500,
        batch_size: int = 32,
        random_state: int = 42,
    ) -> None:
        if min(input_size, hidden_size, epochs, batch_size) <= 0:
            raise ValueError("Network sizes, epochs, and batch size must be positive")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.rng = np.random.default_rng(random_state)
        self.w1 = self.rng.normal(0, 1 / np.sqrt(input_size), (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = self.rng.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, 1))
        self.b2 = np.zeros((1, 1))
        self.loss_history_: list[float] = []

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        clipped = np.clip(value, -40, 40)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = self._sigmoid(features @ self.w1 + self.b1)
        output = hidden @ self.w2 + self.b2
        return hidden, output

    def fit(self, features: np.ndarray, target: np.ndarray) -> "BPRegressor":
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float).reshape(-1, 1)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("Features must be 2-D and aligned with the target")
        for _ in range(self.epochs):
            order = self.rng.permutation(len(x))
            for start in range(0, len(x), self.batch_size):
                batch = order[start : start + self.batch_size]
                xb, yb = x[batch], y[batch]
                hidden, output = self._forward(xb)
                delta_output = 2.0 * (output - yb) / len(xb)
                grad_w2 = hidden.T @ delta_output
                grad_b2 = delta_output.sum(axis=0, keepdims=True)
                delta_hidden = (delta_output @ self.w2.T) * hidden * (1.0 - hidden)
                grad_w1 = xb.T @ delta_hidden
                grad_b1 = delta_hidden.sum(axis=0, keepdims=True)
                self.w2 -= self.learning_rate * grad_w2
                self.b2 -= self.learning_rate * grad_b2
                self.w1 -= self.learning_rate * grad_w1
                self.b1 -= self.learning_rate * grad_b1
            prediction = self.predict(x)
            self.loss_history_.append(float(np.mean((prediction - y.ravel()) ** 2)))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.ndim != 2:
            raise ValueError("Features must be a 2-D array")
        return self._forward(x)[1].ravel()
