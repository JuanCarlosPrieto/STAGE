from __future__ import annotations

import numpy as np


def extract_coordinate(positions, axis: int = 0) -> np.ndarray:
    """Extract one coordinate from a trajectory as a one-dimensional array."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 1:
        if axis != 0:
            raise ValueError("axis must be 0 for a one-dimensional trajectory.")
        values = positions
    elif positions.ndim == 2:
        if axis < 0 or axis >= positions.shape[1]:
            raise ValueError(
                f"axis must be between 0 and {positions.shape[1] - 1}."
            )
        values = positions[:, axis]
    else:
        raise ValueError("positions must have shape (N,) or (N, dimension).")
    if not np.all(np.isfinite(values)):
        raise ValueError("positions contain non-finite values.")
    return values.copy()


def weighted_histogram_counts(
    values,
    weights=None,
    bins=30,
    value_range=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute raw histogram mass suitable for replica aggregation."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite one-dimensional array.")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != values.shape:
            raise ValueError("weights must have the same shape as values.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("weights must be finite and non-negative.")

    counts, bin_edges = np.histogram(
        values,
        bins=bins,
        range=value_range,
        weights=weights,
        density=False,
    )
    return counts.astype(float), bin_edges.astype(float)


def normalize_histogram_counts(
    counts,
    bin_edges,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize raw histogram mass into a probability density."""
    counts = np.asarray(counts, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)
    if counts.ndim != 1 or bin_edges.ndim != 1:
        raise ValueError("counts and bin_edges must be one-dimensional.")
    if len(bin_edges) != len(counts) + 1:
        raise ValueError("bin_edges must contain len(counts) + 1 values.")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("counts must be finite and non-negative.")
    widths = np.diff(bin_edges)
    if not np.all(widths > 0.0):
        raise ValueError("bin_edges must be strictly increasing.")

    total_mass = float(np.sum(counts))
    if total_mass <= 0.0:
        raise ValueError("Cannot normalize a histogram with zero mass.")
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    density = counts / (total_mass * widths)
    return centers, density


def weighted_histogram_density(
    values,
    weights=None,
    bins=30,
    value_range=None,
):
    counts, bin_edges = weighted_histogram_counts(
        values=values,
        weights=weights,
        bins=bins,
        value_range=value_range,
    )
    bin_centers, density = normalize_histogram_counts(counts, bin_edges)
    return bin_centers, density, counts, bin_edges


def theoretical_density_1d(potential, x_values, D):
    x_values = np.asarray(x_values, dtype=float)
    if x_values.ndim != 1 or len(x_values) < 2:
        raise ValueError("x_values must be a one-dimensional grid.")
    if not np.all(np.diff(x_values) > 0.0):
        raise ValueError("x_values must be strictly increasing.")
    D = float(D)
    if not np.isfinite(D) or D <= 0.0:
        raise ValueError("D must be finite and strictly positive.")

    energy = np.array(
        [potential.potential_at(np.array([x])) for x in x_values],
        dtype=float,
    )
    unnormalized = np.exp(-(energy - np.min(energy)) / D)
    normalization = float(np.trapezoid(unnormalized, x_values))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("The theoretical density cannot be normalized.")
    return unnormalized / normalization


def theoretical_marginal_2d(
    potential,
    x_values,
    y_values,
    D,
    axis=0,
):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x_values and y_values must be one-dimensional.")
    if len(x_values) < 2 or len(y_values) < 2:
        raise ValueError("Both coordinate grids require at least two points.")
    if not np.all(np.diff(x_values) > 0.0) or not np.all(
        np.diff(y_values) > 0.0
    ):
        raise ValueError("Coordinate grids must be strictly increasing.")
    D = float(D)
    if not np.isfinite(D) or D <= 0.0:
        raise ValueError("D must be finite and strictly positive.")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")

    X, Y = np.meshgrid(x_values, y_values, indexing="xy")
    energy = np.empty_like(X, dtype=float)
    for row in range(X.shape[0]):
        for column in range(X.shape[1]):
            energy[row, column] = potential.potential_at(
                np.array([X[row, column], Y[row, column]])
            )
    density = np.exp(-(energy - np.min(energy)) / D)

    if axis == 0:
        coordinate = x_values
        marginal = np.trapezoid(density, y_values, axis=0)
    else:
        coordinate = y_values
        marginal = np.trapezoid(density, x_values, axis=1)

    normalization = float(np.trapezoid(marginal, coordinate))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("The marginal density cannot be normalized.")
    return coordinate.copy(), marginal / normalization
