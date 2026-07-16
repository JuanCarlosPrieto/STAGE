from __future__ import annotations

import numpy as np

from ._validation import as_position, validate_positive_float


def _prepare_centers(centers, dimension: int) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.size == 0:
        return np.empty((0, dimension), dtype=float)
    centers = centers.reshape(-1, dimension)
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers must contain only finite values.")
    return centers


def gaussian_bias_value(
    position,
    centers,
    height,
    sigma,
    cutoff: float | None = 5.0,
) -> float:
    """Evaluate a sum of isotropic Gaussian bias contributions."""
    position = np.asarray(position, dtype=float).reshape(-1)
    if position.size == 0 or not np.all(np.isfinite(position)):
        raise ValueError("position must be a non-empty finite vector.")
    height = float(height)
    sigma = validate_positive_float("sigma", sigma)
    if not np.isfinite(height):
        raise ValueError("height must be finite.")
    if cutoff is not None:
        cutoff = validate_positive_float("cutoff", cutoff)

    centers = _prepare_centers(centers, position.size)
    if len(centers) == 0 or height == 0.0:
        return 0.0

    differences = position[None, :] - centers
    if cutoff is not None:
        active = np.linalg.norm(differences, axis=1) <= cutoff * sigma
        differences = differences[active]
        if len(differences) == 0:
            return 0.0

    squared_distances = np.einsum("ij,ij->i", differences, differences)
    return float(np.sum(height * np.exp(-0.5 * squared_distances / sigma**2)))


def gaussian_bias_gradient(
    position,
    centers,
    height,
    sigma,
    cutoff: float | None = 5.0,
) -> np.ndarray:
    """Evaluate the gradient of a Gaussian bias sum."""
    position = np.asarray(position, dtype=float).reshape(-1)
    if position.size == 0 or not np.all(np.isfinite(position)):
        raise ValueError("position must be a non-empty finite vector.")
    height = float(height)
    sigma = validate_positive_float("sigma", sigma)
    if not np.isfinite(height):
        raise ValueError("height must be finite.")
    if cutoff is not None:
        cutoff = validate_positive_float("cutoff", cutoff)

    centers = _prepare_centers(centers, position.size)
    if len(centers) == 0 or height == 0.0:
        return np.zeros_like(position)

    differences = position[None, :] - centers
    if cutoff is not None:
        active = np.linalg.norm(differences, axis=1) <= cutoff * sigma
        differences = differences[active]
        if len(differences) == 0:
            return np.zeros_like(position)

    squared_distances = np.einsum("ij,ij->i", differences, differences)
    values = height * np.exp(-0.5 * squared_distances / sigma**2)
    return -np.sum(differences * values[:, None], axis=0) / sigma**2
