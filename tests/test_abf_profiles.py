import numpy as np

from stage_escape.abp_abf.abf_profiles import reconstruct_abf_profiles


def test_constant_force_reconstructs_linear_free_energy():
    force = np.ones(5)
    edges, free_energy, bias = reconstruct_abf_profiles(force, (0.0, 1.0))
    assert len(edges) == 6
    assert np.all(np.diff(free_energy) > 0.0)
    assert np.isclose(np.min(free_energy), 0.0)
    assert np.isclose(np.min(bias), 0.0)
    np.testing.assert_allclose(bias, np.max(free_energy) - free_energy)
