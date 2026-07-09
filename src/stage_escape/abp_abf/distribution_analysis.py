import numpy as np


def extract_coordinate(positions, dimension=1, axis=0):
    """
    Extract a one-dimensional coordinate from a trajectory.

    Parameters
    ----------
    positions : array-like
        Positions with shape (N,), (N, 1), or (N, d).

    dimension : int
        Dimension of the original process.

    axis : int
        Coordinate to extract when dimension > 1.

    Returns
    -------
    values : ndarray of shape (N,)
        Selected coordinate values.
    """
    positions = np.asarray(positions, dtype=float)

    if positions.ndim == 1:
        return positions

    if positions.ndim != 2:
        raise ValueError("positions must have shape (N,), (N, 1), or (N, d).")

    if dimension == 1:
        return positions[:, 0]

    if axis < 0 or axis >= positions.shape[1]:
        raise ValueError(f"axis must be between 0 and {positions.shape[1] - 1}.")

    return positions[:, axis]


def weighted_histogram_counts(values, weights=None, bins=30, value_range=None):
    """
    Compute weighted histogram counts without normalizing.

    This function is useful for parallelization because raw counts can be
    summed across independent simulations.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")

    if weights is not None:
        weights = np.asarray(weights, dtype=float)

        if weights.shape != values.shape:
            raise ValueError(
                f"weights must have the same shape as values. "
                f"Got weights {weights.shape} and values {values.shape}."
            )

    counts, bin_edges = np.histogram(
        values,
        bins=bins,
        range=value_range,
        weights=weights,
        density=False,
    )

    return counts.astype(float), bin_edges


def normalize_histogram_counts(counts, bin_edges):
    """
    Normalize raw histogram counts into a probability density.
    """
    counts = np.asarray(counts, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)

    bin_widths = np.diff(bin_edges)
    total_mass = np.sum(counts)

    if total_mass <= 0:
        raise ValueError("Cannot normalize histogram with zero total mass.")

    density = counts / (total_mass * bin_widths)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_centers, density


def weighted_histogram_density(values, weights=None, bins=30, value_range=None):
    """
    Compute a normalized weighted histogram density.
    """
    counts, bin_edges = weighted_histogram_counts(
        values=values,
        weights=weights,
        bins=bins,
        value_range=value_range,
    )

    bin_centers, density = normalize_histogram_counts(counts, bin_edges)

    return bin_centers, density, counts, bin_edges