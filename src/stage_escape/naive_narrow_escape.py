from __future__ import annotations

import warnings

import numpy as np

from ._narrow_escape_base import NarrowEscapeBase
from .narrow_escape_result import NarrowEscapeResult


class NaiveNarrowEscape(NarrowEscapeBase):
    """Direct narrow-escape simulation with rejected reflective crossings."""

    def run(self, max_steps=1_000_000, *, reset: bool = False) -> NarrowEscapeResult:
        if isinstance(max_steps, bool) or not isinstance(
            max_steps, (int, np.integer)
        ):
            raise TypeError("max_steps must be an integer.")
        max_steps = int(max_steps)
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if reset:
            self._reset_state()

        initial_steps = self.brownian_motion.n_steps
        remaining = max(max_steps - initial_steps, 0)
        for local_step in range(1, remaining + 1):
            a = np.asarray(self.brownian_motion.positions[-1], dtype=float).copy()
            b = self.brownian_motion.step()
            attempted_steps = initial_steps + local_step
            if self.surface.is_inside(b):
                continue

            crossing = self.surface.first_boundary_intersection(a, b)
            escape_index = self.escape_index(crossing.point)
            if escape_index is not None:
                self._replace_last_position(crossing.point)
                escape_time = (
                    attempted_steps - 1 + crossing.theta
                ) * self.brownian_motion.delta_t
                return self._result(
                    attempted_steps=attempted_steps,
                    rejected_steps=self._rejected_steps,
                    escape_index=escape_index,
                    escape_point=crossing.point,
                    escape_time=escape_time,
                )

            # A reflective proposal still consumes one time step. Replacing the
            # invalid endpoint with the previous state records that rejection
            # without altering trajectory length or stochastic time.
            self._replace_last_position(a)
            self._rejected_steps += 1

        return self._result(
            attempted_steps=self.brownian_motion.n_steps,
            rejected_steps=self._rejected_steps,
        )

    def run_simulation_straight_exit(self, max_steps=1_000_000):
        """Legacy tuple-returning wrapper. Prefer :meth:`run`."""
        warnings.warn(
            "run_simulation_straight_exit is deprecated; use run() and inspect "
            "NarrowEscapeResult.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.run(max_steps=max_steps)
        return result.escape_point, result.escape_time

    def plot_narrow_escape_problem(
        self,
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5),
        point_stride=1000,
        point_size=50,
        point_alpha=0.8,
        show_points=True,
        show=False,
    ):
        from .visualization import plot_narrow_escape_problem

        return plot_narrow_escape_problem(
            path=self.brownian_motion.positions,
            surface=self.surface,
            escapes=self.escapes,
            escape_checker=self.check_escape,
            xlim=xlim,
            ylim=ylim,
            point_stride=point_stride,
            point_size=point_size,
            point_alpha=point_alpha,
            show_points=show_points,
            show=show,
        )
