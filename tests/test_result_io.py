import json

import numpy as np

from stage_escape.abp_abf import ABFRealTime, ABPMetaDynamics
from stage_escape.result_io import load_result, save_result


def test_abp_roundtrip(tmp_path, zero_potential_1d):
    result = ABPMetaDynamics(
        deposition_stride=3,
        transition_detector=None,
        delta_t=1e-3,
        D=0.2,
        b=zero_potential_1d,
        sigma=0.2,
        seed=123,
    ).run(max_steps=8)
    stem = tmp_path / "abp_result"
    save_result(result, stem)
    loaded = load_result(stem)
    np.testing.assert_allclose(loaded.positions, result.positions)
    np.testing.assert_allclose(loaded.weights, result.weights)
    assert loaded.termination_reason == result.termination_reason


def test_abf_roundtrip_and_metadata_contains_termination_reason(
    tmp_path, quadratic_potential_1d
):
    result = ABFRealTime(
        transition_detector=None,
        delta_t=1e-3,
        D=0.2,
        initial_position=[0.25],
        b=quadratic_potential_1d,
        bins=20,
        value_range=(-10.0, 10.0),
        seed=123,
    ).run(max_steps=8)
    stem = tmp_path / "abf_result"
    json_path, _ = save_result(result, stem)
    metadata = json.loads(json_path.read_text())
    assert metadata["termination_reason"] == "max_steps"

    loaded = load_result(stem)
    np.testing.assert_allclose(loaded.force_bias, result.force_bias)
    np.testing.assert_array_equal(loaded.visit_counts, result.visit_counts)
