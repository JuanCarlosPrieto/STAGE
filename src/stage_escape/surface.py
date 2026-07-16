from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .geometry_utilities import find_intersection


@dataclass(frozen=True, slots=True)
class BoundaryIntersection:
    surface_index: int
    point: np.ndarray
    theta: float


class Surface:
    """Domain represented as the intersection of inequalities ``phi_i <= 0``."""

    def __init__(self, name, functions: Iterable) -> None:
        self.name = str(name)
        self.functions = tuple(functions)
        if not self.functions:
            raise ValueError("At least one surface function must be provided.")
        if not all(callable(function) for function in self.functions):
            raise TypeError("Every surface function must be callable.")

    @staticmethod
    def _scalar_value(function, point) -> float:
        value = np.asarray(function(point), dtype=float)
        if value.size != 1:
            raise ValueError("Surface functions must return scalar values.")
        result = float(value.reshape(-1)[0])
        if not np.isfinite(result):
            raise ValueError("Surface functions must return finite values.")
        return result

    def values(self, point) -> np.ndarray:
        point = np.asarray(point, dtype=float).reshape(-1)
        if point.size == 0 or not np.all(np.isfinite(point)):
            raise ValueError("point must be a non-empty finite vector.")
        return np.array(
            [self._scalar_value(function, point) for function in self.functions]
        )

    def is_inside(self, point, *, tolerance: float = 0.0) -> bool:
        tolerance = float(tolerance)
        if tolerance < 0.0 or not np.isfinite(tolerance):
            raise ValueError("tolerance must be finite and non-negative.")
        return bool(np.all(self.values(point) <= tolerance))

    def violated_indices(self, point, *, tolerance: float = 0.0) -> tuple[int, ...]:
        return tuple(np.flatnonzero(self.values(point) > tolerance).tolist())

    def exit_surface(self, point):
        """Return the first violated inequality index, preserving the old API."""
        indices = self.violated_indices(point)
        return indices[0] if indices else None

    def first_boundary_intersection(
        self,
        a,
        b,
        *,
        tolerance: float = 1e-10,
    ) -> BoundaryIntersection:
        """Return the earliest admissible outward crossing on ``[a, b]``."""
        a = np.asarray(a, dtype=float).reshape(-1)
        b = np.asarray(b, dtype=float).reshape(-1)
        if a.shape != b.shape:
            raise ValueError("a and b must have the same shape.")
        if not self.is_inside(a, tolerance=tolerance):
            raise ValueError("The segment must start inside the domain.")
        candidates: list[BoundaryIntersection] = []
        for index, function in enumerate(self.functions):
            value_a = self._scalar_value(function, a)
            value_b = self._scalar_value(function, b)
            if value_a > tolerance or value_b <= tolerance:
                continue
            point, theta = find_intersection(function, a, b)
            if self.is_inside(point, tolerance=tolerance):
                candidates.append(
                    BoundaryIntersection(index, point.copy(), float(theta))
                )
        if not candidates:
            raise RuntimeError(
                "The proposed segment leaves the domain but no admissible "
                "boundary intersection could be located."
            )
        return min(candidates, key=lambda candidate: candidate.theta)

    def boundary_segments_2d(self, xlim, ylim, n_grid=400, tol=1e-8):
        """Extract approximate boundary line segments for visualization."""
        if n_grid < 2:
            raise ValueError("n_grid must be at least 2.")
        import matplotlib.pyplot as plt

        x = np.linspace(xlim[0], xlim[1], n_grid)
        y = np.linspace(ylim[0], ylim[1], n_grid)
        X, Y = np.meshgrid(x, y)
        all_segments = []
        fig_tmp, ax_tmp = plt.subplots()
        try:
            for index, function in enumerate(self.functions):
                Z = np.fromiter(
                    (
                        self._scalar_value(function, (x_value, y_value))
                        for x_value, y_value in zip(
                            X.ravel(), Y.ravel(), strict=True
                        )
                    ),
                    dtype=float,
                    count=X.size,
                ).reshape(X.shape)
                contour = ax_tmp.contour(X, Y, Z, levels=[0.0])
                for curve in contour.allsegs[0]:
                    for p_a, p_b in pairwise(curve):
                        midpoint = 0.5 * (p_a + p_b)
                        valid = all(
                            other_index == index
                            or self._scalar_value(other, midpoint) <= tol
                            for other_index, other in enumerate(self.functions)
                        )
                        if valid:
                            all_segments.append(np.array([p_a, p_b]))
        finally:
            plt.close(fig_tmp)
        return all_segments

    def plot_boundary_2d(
        self,
        ax,
        xlim,
        ylim,
        escape_checker=None,
        n_grid=400,
        reflective_color="black",
        escape_color="orange",
        reflective_linewidth=2.0,
        escape_linewidth=4.0,
    ):
        from matplotlib.collections import LineCollection

        segments = self.boundary_segments_2d(xlim=xlim, ylim=ylim, n_grid=n_grid)
        reflective_segments = []
        escape_segments = []
        for segment in segments:
            midpoint = 0.5 * (segment[0] + segment[1])
            if escape_checker is not None and escape_checker(midpoint):
                escape_segments.append(segment)
            else:
                reflective_segments.append(segment)
        artists = []
        if reflective_segments:
            artist = LineCollection(
                reflective_segments,
                colors=reflective_color,
                linewidths=reflective_linewidth,
                label="Reflective boundary",
            )
            ax.add_collection(artist)
            artists.append(artist)
        if escape_segments:
            artist = LineCollection(
                escape_segments,
                colors=escape_color,
                linewidths=escape_linewidth,
                label="Escape window",
            )
            ax.add_collection(artist)
            artists.append(artist)
        return artists
