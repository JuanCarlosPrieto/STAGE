from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

TerminationReason = Literal[
    "transition",
    "max_steps",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulationResult:
    """Information shared by ABP and ABF simulations."""

    method: str
    positions: np.ndarray
    delta_t: float
    diffusion: float
    seed: int | None
    transition_index: int | None
    physical_time: float
    metadata: dict[str, Any] = field(default_factory=dict)
    termination_reason: TerminationReason

    @property
    def n_steps(self) -> int:
        return max(len(self.positions) - 1, 0)

    @property
    def simulated_time(self) -> float:
        """Time measured in the biased simulation clock."""
        return self.n_steps * self.delta_t

    @property
    def transition_detected(self) -> bool:
        return self.transition_index is not None

    @property
    def biased_transition_time(self) -> float | None:
        if self.transition_index is None:
            return None

        return self.transition_index * self.delta_t


@dataclass(frozen=True, slots=True, kw_only=True)
class ABPResult(SimulationResult):
    """Complete output of an ABP simulation."""

    weights: np.ndarray
    centers: np.ndarray
    bias_height: float
    bias_width: float


@dataclass(frozen=True, slots=True, kw_only=True)
class ABFResult(SimulationResult):
    """Complete output of a one-dimensional ABF simulation."""

    force_bias: np.ndarray
    visit_counts: np.ndarray
    bin_edges: np.ndarray
    free_energy: np.ndarray
    bias_potential: np.ndarray

    @property
    def bin_centers(self) -> np.ndarray:
        return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])