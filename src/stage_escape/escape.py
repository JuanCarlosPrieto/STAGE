from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class Escape:
    """Conjunction of geometric conditions defining an escape window."""

    def __init__(self, conditions: Iterable) -> None:
        self.conditions = tuple(conditions)
        if not self.conditions:
            raise ValueError("At least one escape condition must be provided.")
        if not all(callable(condition) for condition in self.conditions):
            raise TypeError("Every escape condition must be callable.")

    def is_valid_escape(self, point) -> bool:
        point = np.asarray(point, dtype=float).reshape(-1)
        if point.size == 0 or not np.all(np.isfinite(point)):
            raise ValueError("point must be a non-empty finite vector.")
        for condition in self.conditions:
            value = np.asarray(condition(point))
            if value.size != 1:
                raise ValueError("Escape conditions must return scalar values.")
            if not bool(value.reshape(-1)[0]):
                return False
        return True

    def plot_escape(
        self,
        ax,
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5),
        n_grid=400,
        escape_color="orange",
        region_alpha=0.25,
    ):
        """Paint the 2D region satisfying all conditions and return the artist."""
        if n_grid < 2:
            raise ValueError("n_grid must be at least 2.")
        x = np.linspace(xlim[0], xlim[1], n_grid)
        y = np.linspace(ylim[0], ylim[1], n_grid)
        X, Y = np.meshgrid(x, y)
        mask = np.fromiter(
            (
                self.is_valid_escape((x_value, y_value))
                for x_value, y_value in zip(X.ravel(), Y.ravel(), strict=True)
            ),
            dtype=bool,
            count=X.size,
        ).reshape(X.shape)
        artist = ax.contourf(
            X,
            Y,
            mask.astype(float),
            levels=[0.5, 1.5],
            colors=[escape_color],
            alpha=region_alpha,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        return artist
