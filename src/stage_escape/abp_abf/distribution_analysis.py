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


def theoretical_density_1d(potential, x_values, D):
    x_values = np.asarray(x_values, dtype=float)

    if D <= 0:
        raise ValueError("D must be strictly positive.")

    energy = np.array(
        [
            potential.potential_at(np.array([x], dtype=float))
            for x in x_values
        ],
        dtype=float,
    )

    unnormalized = np.exp(-(energy - energy.min()) / D)
    normalization = np.trapezoid(unnormalized, x_values)

    if not np.isfinite(normalization) or normalization <= 0:
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

    if D <= 0:
        raise ValueError("D must be strictly positive.")

    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")

    X, Y = np.meshgrid(x_values, y_values, indexing="xy")

    energy = np.empty_like(X, dtype=float)

    for row in range(X.shape[0]):
        for col in range(X.shape[1]):
            energy[row, col] = potential.potential_at(
                np.array([X[row, col], Y[row, col]])
            )

    density = np.exp(-(energy - energy.min()) / D)

    if axis == 0:
        coordinate = x_values
        marginal = np.trapezoid(density, y_values, axis=0)
    else:
        coordinate = y_values
        marginal = np.trapezoid(density, x_values, axis=1)

    normalization = np.trapezoid(marginal, coordinate)

    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("The marginal density cannot be normalized.")

    return coordinate, marginal / normalization


def _evaluate_potential_on_grid(
    self,
    potential_type,
    x_range=(-2, 2),
    y_range=(-2, 2),
    num_points=100
):
    """
    Evaluate one of the potentials on a 2D grid.

    potential_type can be:
    - "original"
    - "biasing"
    - "final"
    """

    x_values = np.linspace(x_range[0], x_range[1], num_points)
    y_values = np.linspace(y_range[0], y_range[1], num_points)

    X, Y = np.meshgrid(x_values, y_values)
    Z = np.zeros_like(X, dtype=float)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])

            if potential_type == "original":
                Z[i, j] = self.b.potential_at(point)

            elif potential_type == "biasing":
                Z[i, j] = self.bias_potential_at(point)

            elif potential_type == "final":
                Z[i, j] = (
                    self.b.potential_at(point)
                    + self.bias_potential_at(point)
                )

            else:
                raise ValueError(
                    "potential_type must be 'original', 'biasing', or 'final'"
                )

    return X, Y, Z