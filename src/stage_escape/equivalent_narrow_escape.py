from __future__ import annotations

import warnings

import numpy as np

from ._narrow_escape_base import NarrowEscapeBase
from .narrow_escape_result import NarrowEscapeResult


class EquivalentNarrowEscape(NarrowEscapeBase):
    """Equivalent escape simulation using endpoint escape regions."""

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
            previous = np.asarray(
                self.brownian_motion.positions[-1], dtype=float
            ).copy()
            point = self.brownian_motion.step()
            attempted_steps = initial_steps + local_step
            escape_index = self.escape_index(point)
            if escape_index is not None:
                return self._result(
                    attempted_steps=attempted_steps,
                    rejected_steps=self._rejected_steps,
                    escape_index=escape_index,
                    escape_point=point,
                    escape_time=attempted_steps * self.brownian_motion.delta_t,
                )
            if not self.surface.is_inside(point):
                self._replace_last_position(previous)
                self._rejected_steps += 1

        return self._result(
            attempted_steps=self.brownian_motion.n_steps,
            rejected_steps=self._rejected_steps,
        )

    def run_simulation(self, max_steps=1_000_000):
        """Legacy wrapper preserving the historical dictionary interface."""
        warnings.warn(
            "run_simulation is deprecated; use run() and inspect "
            "NarrowEscapeResult.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.run(max_steps=max_steps)
        if not result.escaped:
            return None, None
        return {
            "escape_index": result.escape_index,
            "escape_point": result.escape_point,
            "escape_time": result.escape_time,
        }

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
            escape_checker=None,
            xlim=xlim,
            ylim=ylim,
            point_stride=point_stride,
            point_size=point_size,
            point_alpha=point_alpha,
            show_points=show_points,
            show=show,
        )
