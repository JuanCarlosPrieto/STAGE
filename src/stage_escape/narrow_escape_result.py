from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

NarrowEscapeTermination = Literal["escape", "max_steps"]


@dataclass(frozen=True, slots=True, kw_only=True)
class NarrowEscapeResult:
    """Immutable output shared by direct and equivalent escape simulations."""

    positions: np.ndarray
    delta_t: float
    attempted_steps: int
    rejected_steps: int
    termination_reason: NarrowEscapeTermination
    escape_index: int | None = None
    escape_point: np.ndarray | None = None
    escape_time: float | None = None

    def __post_init__(self) -> None:
        positions = np.array(self.positions, dtype=float, copy=True)
        if positions.ndim != 2 or not positions.size:
            raise ValueError("positions must have shape (n_points, dimension).")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions must contain finite values.")
        positions.setflags(write=False)

        delta_t = float(self.delta_t)
        if not np.isfinite(delta_t) or delta_t <= 0.0:
            raise ValueError("delta_t must be finite and strictly positive.")
        if isinstance(self.attempted_steps, bool) or not isinstance(
            self.attempted_steps, (int, np.integer)
        ):
            raise TypeError("attempted_steps must be an integer.")
        if isinstance(self.rejected_steps, bool) or not isinstance(
            self.rejected_steps, (int, np.integer)
        ):
            raise TypeError("rejected_steps must be an integer.")
        attempted_steps = int(self.attempted_steps)
        rejected_steps = int(self.rejected_steps)
        if attempted_steps < 0:
            raise ValueError("attempted_steps must be non-negative.")
        if rejected_steps < 0:
            raise ValueError("rejected_steps must be non-negative.")
        if rejected_steps > attempted_steps:
            raise ValueError("rejected_steps cannot exceed attempted_steps.")
        if self.termination_reason not in {"escape", "max_steps"}:
            raise ValueError("Invalid termination_reason.")

        escape_index = self.escape_index
        escape_point = self.escape_point
        escape_time = self.escape_time
        if self.termination_reason == "escape":
            if escape_index is None or escape_point is None or escape_time is None:
                raise ValueError("Escape termination requires complete escape data.")
            if isinstance(escape_index, bool) or int(escape_index) < 0:
                raise ValueError("escape_index must be a non-negative integer.")
            escape_index = int(escape_index)
            escape_point = np.array(escape_point, dtype=float, copy=True).reshape(-1)
            if escape_point.shape != (positions.shape[1],):
                raise ValueError("escape_point must have the trajectory dimension.")
            if not np.all(np.isfinite(escape_point)):
                raise ValueError("escape_point must contain finite values.")
            escape_point.setflags(write=False)
            escape_time = float(escape_time)
            if not np.isfinite(escape_time) or escape_time < 0.0:
                raise ValueError("escape_time must be finite and non-negative.")
        elif any(value is not None for value in (escape_index, escape_point, escape_time)):
            raise ValueError("max_steps termination cannot contain escape data.")

        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "delta_t", delta_t)
        object.__setattr__(self, "attempted_steps", attempted_steps)
        object.__setattr__(self, "rejected_steps", rejected_steps)
        object.__setattr__(self, "escape_index", escape_index)
        object.__setattr__(self, "escape_point", escape_point)
        object.__setattr__(self, "escape_time", escape_time)

    @property
    def escaped(self) -> bool:
        return self.termination_reason == "escape"

    @property
    def dimension(self) -> int:
        return int(self.positions.shape[1])
