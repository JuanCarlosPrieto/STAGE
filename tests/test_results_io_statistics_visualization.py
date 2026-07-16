import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from stage_escape.abp_abf import ABPMetaDynamics, Potential, TransitionDetector
from stage_escape.result_io import load_result, save_result
from stage_escape.statistics import Statistics
from stage_escape.visualization import plot_brownian_path


def abp_result():
    potential = Potential(
        1,
        lambda x: 0.5 * x[0] ** 2,
        lambda x: np.array([x[0]]),
    )
    detector = TransitionDetector(lambda x: x[0], threshold=np.inf)
    return ABPMetaDynamics(
        3, detector, 1e-3, D=0.2, b=potential, W=0.01, sigma=0.2, seed=2
    ).run(8)


def test_result_metadata_and_arrays_are_immutable():
    result = abp_result()
    with pytest.raises(TypeError):
        result.metadata["new"] = 1
    with pytest.raises(ValueError):
        result.positions[0, 0] = 1


def test_result_roundtrip(tmp_path):
    original = abp_result()
    json_path, npz_path = save_result(original, tmp_path / "case")
    assert json_path.exists() and npz_path.exists()
    loaded = load_result(tmp_path / "case")
    np.testing.assert_allclose(loaded.positions, original.positions)
    np.testing.assert_allclose(loaded.weights, original.weights)
    assert dict(loaded.metadata) == dict(original.metadata)


def test_unknown_schema_is_rejected(tmp_path):
    original = abp_result()
    json_path, _ = save_result(original, tmp_path / "case")
    metadata = json.loads(json_path.read_text())
    metadata["schema_version"] = 999
    json_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Unsupported"):
        load_result(tmp_path / "case")


def test_exponential_rate_estimator():
    assert Statistics.adapt_exp_distribution([1, 2, 3], t_min=1) == pytest.approx(1.0)


def test_plotting_does_not_show_by_default(monkeypatch):
    called = False

    def fake_show():
        nonlocal called
        called = True

    monkeypatch.setattr(plt, "show", fake_show)
    fig, ax = plot_brownian_path([[0.0], [1.0], [0.5]], mode_1d="time_series")
    assert fig is ax.figure
    assert not called
    plt.close(fig)
