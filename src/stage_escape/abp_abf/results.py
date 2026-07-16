from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

TerminationReason = Literal["transition", "max_steps"]


def _readonly_array(
    value,
    *,
    dtype=float,
    ndim: int | None = None,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"Expected an array with {ndim} dimensions.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Result arrays must contain only finite values.")
    array.setflags(write=False)
    return array


def _readonly_integer_array(value, *, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise ValueError(f"Expected an array with {ndim} dimensions.")
    if not np.issubdtype(raw.dtype, np.integer):
        numeric = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise ValueError("Integer result arrays must contain integer values.")
    return _readonly_array(value, dtype=np.int64, ndim=ndim)


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
    termination_reason: TerminationReason
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positions = _readonly_array(self.positions, ndim=2)
        if positions.shape[0] == 0 or positions.shape[1] == 0:
            raise ValueError("positions must have shape (n_points, dimension).")

        delta_t = float(self.delta_t)
        diffusion = float(self.diffusion)
        physical_time = float(self.physical_time)
        if not np.isfinite(delta_t) or delta_t <= 0.0:
            raise ValueError("delta_t must be finite and strictly positive.")
        if not np.isfinite(diffusion) or diffusion <= 0.0:
            raise ValueError("diffusion must be finite and strictly positive.")
        if not np.isfinite(physical_time) or physical_time < 0.0:
            raise ValueError("physical_time must be finite and non-negative.")
        if self.termination_reason not in {"transition", "max_steps"}:
            raise ValueError("Invalid termination_reason.")

        seed = self.seed
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
                raise TypeError("seed must be an integer or None.")
            seed = int(seed)

        transition_index = self.transition_index
        if transition_index is not None:
            if isinstance(transition_index, bool) or not isinstance(
                transition_index, (int, np.integer)
            ):
                raise TypeError("transition_index must be an integer or None.")
            transition_index = int(transition_index)
            if transition_index < 0 or transition_index >= len(positions):
                raise ValueError("transition_index lies outside the trajectory.")
            if self.termination_reason != "transition":
                raise ValueError(
                    "A non-null transition_index requires "
                    "termination_reason='transition'."
                )
        elif self.termination_reason == "transition":
            raise ValueError(
                "termination_reason='transition' requires transition_index."
            )

        metadata = MappingProxyType(deepcopy(dict(self.metadata)))
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "delta_t", delta_t)
        object.__setattr__(self, "diffusion", diffusion)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "physical_time", physical_time)
        object.__setattr__(self, "transition_index", transition_index)
        object.__setattr__(self, "metadata", metadata)

    @property
    def dimension(self) -> int:
        return int(self.positions.shape[1])

    @property
    def n_steps(self) -> int:
        return max(len(self.positions) - 1, 0)

    @property
    def simulated_time(self) -> float:
        """Elapsed time in the biased simulation clock."""
        return self.n_steps * self.delta_t

    @property
    def transition_detected(self) -> bool:
        return self.transition_index is not None

    @property
    def biased_transition_time(self) -> float | None:
        if self.transition_index is None:
            return None
        return self.transition_index * self.delta_t

    @property
    def reweighted_time(self) -> float:
        return self.physical_time


@dataclass(frozen=True, slots=True, kw_only=True)
class ABPResult(SimulationResult):
    """Complete output of an ABP/metadynamics simulation."""

    weights: np.ndarray
    centers: np.ndarray
    bias_height: float
    bias_width: float

    def __post_init__(self) -> None:
        super(ABPResult, self).__post_init__()
        if self.method != "abp":
            raise ValueError("ABPResult requires method='abp'.")
        weights = _readonly_array(self.weights, ndim=1)
        centers = _readonly_array(self.centers, ndim=2)
        if len(weights) != len(self.positions):
            raise ValueError("weights and positions must have the same length.")
        if np.any(weights <= 0.0):
            raise ValueError("ABP weights must be strictly positive.")
        if centers.shape[1] != self.dimension:
            raise ValueError("centers must have the trajectory dimension.")

        bias_height = float(self.bias_height)
        bias_width = float(self.bias_width)
        if not np.isfinite(bias_height) or bias_height < 0.0:
            raise ValueError("bias_height must be finite and non-negative.")
        if not np.isfinite(bias_width) or bias_width <= 0.0:
            raise ValueError("bias_width must be finite and strictly positive.")

        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "bias_height", bias_height)
        object.__setattr__(self, "bias_width", bias_width)


@dataclass(frozen=True, slots=True, kw_only=True)
class ABFResult(SimulationResult):
    """Complete output of a one-dimensional ABF simulation."""

    force_bias: np.ndarray
    visit_counts: np.ndarray
    bin_edges: np.ndarray
    free_energy: np.ndarray
    bias_potential: np.ndarray

    def __post_init__(self) -> None:
        super(ABFResult, self).__post_init__()
        if self.method != "abf":
            raise ValueError("ABFResult requires method='abf'.")
        if self.dimension != 1:
            raise ValueError("ABFResult only supports one-dimensional paths.")

        force_bias = _readonly_array(self.force_bias, ndim=1)
        visit_counts = _readonly_integer_array(self.visit_counts, ndim=1)
        bin_edges = _readonly_array(self.bin_edges, ndim=1)
        free_energy = _readonly_array(self.free_energy, ndim=1)
        bias_potential = _readonly_array(self.bias_potential, ndim=1)

        n_bins = len(force_bias)
        if n_bins < 2:
            raise ValueError("ABF results require at least two bins.")
        if len(visit_counts) != n_bins:
            raise ValueError("visit_counts must match force_bias.")
        if len(bin_edges) != n_bins + 1:
            raise ValueError("bin_edges must contain n_bins + 1 values.")
        if len(free_energy) != n_bins or len(bias_potential) != n_bins:
            raise ValueError("ABF profile arrays must match force_bias.")
        if np.any(visit_counts < 0):
            raise ValueError("visit_counts must be non-negative.")
        if not np.all(np.diff(bin_edges) > 0.0):
            raise ValueError("bin_edges must be strictly increasing.")

        object.__setattr__(self, "force_bias", force_bias)
        object.__setattr__(self, "visit_counts", visit_counts)
        object.__setattr__(self, "bin_edges", bin_edges)
        object.__setattr__(self, "free_energy", free_energy)
        object.__setattr__(self, "bias_potential", bias_potential)

    @property
    def bin_centers(self) -> np.ndarray:
        centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        centers.setflags(write=False)
        return centers

# Explicit name used by the public API; SimulationResult remains compatible.
AdaptiveSimulationResult = SimulationResult
