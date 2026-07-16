import numpy as np


def gaussian_bias_value(
    position,
    centers,
    height,
    sigma,
    cutoff=5.0,
):
    position = np.asarray(
        position,
        dtype=float,
    ).reshape(-1)

    centers = np.asarray(
        centers,
        dtype=float,
    )

    if sigma <= 0:
        raise ValueError("sigma must be strictly positive.")

    if centers.size == 0:
        return 0.0

    centers = centers.reshape(
        -1,
        position.size,
    )

    differences = position[None, :] - centers

    active = (
        np.max(
            np.abs(differences),
            axis=1,
        )
        < cutoff * sigma
    )

    if not np.any(active):
        return 0.0

    squared_distances = np.sum(
        differences[active] ** 2,
        axis=1,
    )

    contributions = height * np.exp(
        -0.5
        * squared_distances
        / sigma**2
    )

    return float(np.sum(contributions))


def gaussian_bias_gradient(
    position,
    centers,
    height,
    sigma,
    cutoff=5.0,
):
    position = np.asarray(
        position,
        dtype=float,
    ).reshape(-1)

    centers = np.asarray(
        centers,
        dtype=float,
    )

    if sigma <= 0:
        raise ValueError("sigma must be strictly positive.")

    if centers.size == 0:
        return np.zeros_like(position)

    centers = centers.reshape(
        -1,
        position.size,
    )

    differences = position[None, :] - centers

    active = (
        np.max(
            np.abs(differences),
            axis=1,
        )
        < cutoff * sigma
    )

    if not np.any(active):
        return np.zeros_like(position)

    active_differences = differences[active]

    gaussian_values = np.exp(
        -0.5
        * np.sum(
            active_differences**2,
            axis=1,
        )
        / sigma**2
    )

    gradient = (
        -height
        / sigma**2
        * np.sum(
            active_differences
            * gaussian_values[:, None],
            axis=0,
        )
    )

    return gradient