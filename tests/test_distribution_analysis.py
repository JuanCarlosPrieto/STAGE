import numpy as np

from stage_escape.abp_abf import (
    extract_coordinate,
    normalize_histogram_counts,
    weighted_histogram_counts,
)


def test_histogram_counts_can_be_aggregated_before_normalization():
    values_1 = np.array([-0.8, -0.2, 0.3])
    values_2 = np.array([-0.4, 0.5, 0.9])
    counts_1, edges = weighted_histogram_counts(
        values_1, bins=4, value_range=(-1.0, 1.0)
    )
    counts_2, edges_2 = weighted_histogram_counts(
        values_2, bins=4, value_range=(-1.0, 1.0)
    )
    np.testing.assert_allclose(edges, edges_2)
    centers, density = normalize_histogram_counts(counts_1 + counts_2, edges)
    assert len(centers) == len(density) == 4
    assert np.isclose(np.sum(density * np.diff(edges)), 1.0)


def test_extract_coordinate_from_matrix():
    positions = np.array([[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_allclose(extract_coordinate(positions, axis=1), [1.0, 3.0])
