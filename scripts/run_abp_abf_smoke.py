"""Small deterministic smoke run for the refactored ABP/ABF API."""

from pathlib import Path

import numpy as np

from stage_escape.abp_abf import (
    ABFRealTime,
    ABPMetaDynamics,
    Potential,
    TransitionDetector,
)
from stage_escape.result_io import save_result


def quadratic_potential() -> Potential:
    return Potential(
        dimension=1,
        function=lambda x: 0.5 * float(x[0]) ** 2,
        first_derivative=lambda x: np.array([float(x[0])]),
        second_derivative=lambda x: np.array([[1.0]]),
    )


def main() -> None:
    output = Path("results/smoke")
    detector = TransitionDetector(lambda x: x[0], threshold=1.0)
    potential = quadratic_potential()

    abp = ABPMetaDynamics(
        deposition_stride=50,
        transition_detector=detector,
        delta_t=1e-3,
        D=0.2,
        initial_position=[0.0],
        b=potential,
        W=0.05,
        sigma=0.2,
        seed=123,
    )
    save_result(abp.run(max_steps=2_000), output / "abp")

    abf = ABFRealTime(
        transition_detector=detector,
        delta_t=1e-3,
        D=0.2,
        initial_position=[0.0],
        b=potential,
        bins=100,
        value_range=(-3.0, 3.0),
        seed=123,
    )
    save_result(abf.run(max_steps=2_000), output / "abf")


if __name__ == "__main__":
    main()
