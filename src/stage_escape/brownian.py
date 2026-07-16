from __future__ import annotations

from collections.abc import Callable

import numpy as np


def zero_drift(x: np.ndarray) -> np.ndarray:
    """Default drift field ``b(x) = 0``."""
    return np.zeros_like(x, dtype=float)


class BrownianMotion:
    """Euler-Maruyama simulation with optional drift.

    The trajectory remains stored in ``positions`` for compatibility with the
    narrow-escape simulators. ``step`` advances exactly once. ``simulate``
    preserves the historical block convention: a block size of ``n`` appends
    ``n - 1`` new positions because the current point is the first block point.
    """

    def __init__(
        self,
        deposition_stride,
        delta_t,
        dimension=1,
        D=1.0,
        initial_position=None,
        b: Callable[[np.ndarray], np.ndarray] = zero_drift,
        seed=None,
        rng=None,
    ) -> None:
        if isinstance(deposition_stride, bool):
            raise TypeError("deposition_stride must be an integer.")
        self.deposition_stride = int(deposition_stride)
        self.delta_t = float(delta_t)
        self.dimension = int(dimension)
        self.D = float(D)
        if not callable(b):
            raise TypeError("b must be callable.")
        self.b = b

        if self.deposition_stride < 2:
            raise ValueError("deposition_stride must be at least 2.")
        if not np.isfinite(self.delta_t) or self.delta_t <= 0.0:
            raise ValueError("delta_t must be finite and positive.")
        if self.dimension < 1:
            raise ValueError("dimension must be at least 1.")
        if not np.isfinite(self.D) or self.D < 0.0:
            raise ValueError("D must be finite and non-negative.")

        if initial_position is None:
            initial_position = np.zeros(self.dimension, dtype=float)
        else:
            initial_position = np.asarray(initial_position, dtype=float)
        if initial_position.shape != (self.dimension,):
            raise ValueError(
                f"initial_position must have shape ({self.dimension},), "
                f"got {initial_position.shape}."
            )
        if not np.all(np.isfinite(initial_position)):
            raise ValueError("initial_position must contain finite values.")

        if seed is not None and rng is not None:
            raise ValueError("Provide either seed or rng, not both.")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be an instance of numpy.random.Generator.")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
        ):
            raise TypeError("seed must be an integer or None.")

        self.initial_position = initial_position.copy()
        self.seed = None if seed is None else int(seed)
        self._external_rng = rng is not None
        self.rng = rng if rng is not None else np.random.default_rng(self.seed)
        self.positions = [self.initial_position.copy()]

    @property
    def n_steps(self) -> int:
        return max(len(self.positions) - 1, 0)

    def reset(self, *, reset_rng: bool = True) -> None:
        """Reset the trajectory and, by default, an internally owned RNG."""
        self.positions = [self.initial_position.copy()]
        if reset_rng and not self._external_rng:
            self.rng = np.random.default_rng(self.seed)

    def _drift_at(self, position) -> np.ndarray:
        drift = np.asarray(self.b(position), dtype=float)
        if drift.shape == ():
            drift = np.full(self.dimension, float(drift))
        if drift.shape != (self.dimension,):
            raise ValueError(
                f"Drift must return shape ({self.dimension},), got {drift.shape}."
            )
        if not np.all(np.isfinite(drift)):
            raise FloatingPointError("The drift returned non-finite values.")
        return drift

    def propose(self, position=None) -> np.ndarray:
        """Generate one proposal without mutating the stored trajectory."""
        current = (
            np.asarray(self.positions[-1], dtype=float)
            if position is None
            else np.asarray(position, dtype=float)
        )
        if current.shape != (self.dimension,):
            raise ValueError(
                f"position must have shape ({self.dimension},), got {current.shape}."
            )
        drift = self._drift_at(current) * self.delta_t
        noise = (
            np.sqrt(2.0 * self.D * self.delta_t)
            * self.rng.standard_normal(self.dimension)
        )
        proposal = current + drift + noise
        if not np.all(np.isfinite(proposal)):
            raise FloatingPointError("The Brownian proposal is not finite.")
        return proposal

    def step(self) -> np.ndarray:
        """Advance the trajectory by exactly one integration step."""
        proposal = self.propose()
        self.positions.append(proposal.copy())
        return proposal.copy()

    def simulate(self, deposition_stride=None, reset=False) -> np.ndarray:
        """Extend the path by one historical simulation block."""
        if reset:
            self.reset()
        block_points = (
            self.deposition_stride
            if deposition_stride is None
            else int(deposition_stride)
        )
        if block_points < 2:
            raise ValueError("deposition_stride must be at least 2.")
        for _ in range(block_points - 1):
            self.step()
        return np.asarray(self.positions, dtype=float)

    @staticmethod
    def plot_brownian_path(*args, **kwargs):
        """Backward-compatible wrapper around the visualization module."""
        from .visualization import plot_brownian_path

        return plot_brownian_path(*args, **kwargs)

    @staticmethod
    def set_axes_equal_3d(ax, points):
        from .visualization import set_axes_equal_3d

        return set_axes_equal_3d(ax, points)
