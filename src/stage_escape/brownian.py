# Brownian Motion Simulation

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# src/stage_escape/brownian.py

import numpy as np


def zero_drift(x):
    """
    Default drift field b(x) = 0.

    It returns a vector with the same shape as x.
    """
    return np.zeros_like(x, dtype=float)


class BrownianMotion:
    """
    Brownian motion with optional drift.

    This class intentionally stores the trajectory in self.positions because
    other parts of the project, especially NaiveNarrowEscape and
    EquivalentNarrowEscape, mutate and extend this list.

    Discretization:

        X_{n+1} = X_n + b(X_n) dt + sqrt(2D dt) Z_n

    where Z_n is a standard normal random vector.
    """

    def __init__(
        self,
        num_steps,
        delta_t,
        dimension=1,
        D=1.0,
        initial_position=None,
        b=zero_drift,
        seed=None,
        rng=None,
    ):
        self.num_steps = int(num_steps)
        self.delta_t = float(delta_t)
        self.dimension = int(dimension)
        self.D = float(D)
        self.b = b

        if self.num_steps < 2:
            raise ValueError("num_steps must be at least 2.")

        if self.delta_t <= 0:
            raise ValueError("delta_t must be positive.")

        if self.dimension < 1:
            raise ValueError("dimension must be at least 1.")

        if self.D < 0:
            raise ValueError("D must be non-negative.")

        if initial_position is None:
            initial_position = np.zeros(self.dimension, dtype=float)
        else:
            initial_position = np.asarray(initial_position, dtype=float)

        if initial_position.shape != (self.dimension,):
            raise ValueError(
                f"initial_position must have shape ({self.dimension},), "
                f"got {initial_position.shape}."
            )

        self.initial_position = initial_position.copy()
        self.positions = [self.initial_position.copy()]

        self.seed = seed
        self._external_rng = rng is not None
        self.rng = rng if rng is not None else np.random.default_rng(seed)


    def reset(self):
        """
        Reset the trajectory to the initial position.

        If the random generator was created from a seed, the generator is also
        reset, so the same simulation can be reproduced.
        """
        self.positions = [self.initial_position.copy()]

        if not self._external_rng:
            self.rng = np.random.default_rng(self.seed)


    def _drift_at(self, position):
        """
        Evaluate and validate the drift at the current position.

        Scalar drifts are accepted for backward compatibility and are broadcast
        to all dimensions.
        """
        drift = np.asarray(self.b(position), dtype=float)

        if drift.shape == ():
            drift = np.full(self.dimension, float(drift))

        if drift.shape != (self.dimension,):
            raise ValueError(
                f"Drift must return shape ({self.dimension},), "
                f"got {drift.shape}."
            )

        return drift

    def simulate(self, num_steps=None, reset=False):
        """
        Extend the Brownian trajectory.

        Parameters
        ----------
        num_steps : int, optional
            Number of points to generate in this simulation block. If None,
            self.num_steps is used.

        reset : bool
            If True, reset the trajectory before simulating.

        Returns
        -------
        np.ndarray
            Array version of self.positions.

        Notes
        -----
        By default, this method EXTENDS self.positions instead of replacing it.
        This is necessary for compatibility with the current narrow escape
        simulation classes.
        """
        if reset:
            self.reset()

        n_steps = self.num_steps if num_steps is None else int(num_steps)

        if n_steps < 2:
            raise ValueError("num_steps must be at least 2.")

        step_size = np.sqrt(2.0 * self.D * self.delta_t)

        for _ in range(1, n_steps):
            current_position = self.positions[-1]
            drift = self._drift_at(current_position) * self.delta_t
            noise = step_size * self.rng.standard_normal(self.dimension)

            new_position = current_position + drift + noise
            self.positions.append(new_position)

        return np.asarray(self.positions, dtype=float)

    @staticmethod
    def plot_brownian_path(*args, **kwargs):
        """
        Backward-compatible wrapper.

        The actual plotting function lives in visualization.py, but this method
        is kept so existing notebooks using BrownianMotion.plot_brownian_path(...)
        do not break.
        """
        from .visualization import plot_brownian_path

        return plot_brownian_path(*args, **kwargs)

    @staticmethod
    def set_axes_equal_3d(ax, points):
        """
        Backward-compatible wrapper.
        """
        from .visualization import set_axes_equal_3d

        return set_axes_equal_3d(ax, points)