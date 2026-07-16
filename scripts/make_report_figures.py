from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stage_escape.abp_abf import (
    Potential,
    theoretical_density_1d,
    weighted_histogram_density,
)
from stage_escape.result_io import load_result
from stage_escape.visualization import plot_histogram_vs_distribution


def double_well() -> Potential:
    return Potential(1, lambda x: 0.25 * (x[0] ** 2 - 1.0) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create report-ready figures from a saved result.")
    parser.add_argument("result", type=Path, help="Result stem (.json/.npz suffix optional)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/figures"))
    args = parser.parse_args()

    result = load_result(args.result)
    args.output.mkdir(parents=True, exist_ok=True)
    positions = result.positions[:, 0]
    weights = getattr(result, "weights", np.ones_like(positions))
    centers, density, _, edges = weighted_histogram_density(
        positions, weights, bins=60
    )
    x_theory = np.linspace(edges[0], edges[-1], 1000)
    theory = theoretical_density_1d(double_well(), x_theory, result.diffusion)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_histogram_vs_distribution(ax, centers, density, x_theory, theory)
    fig.tight_layout()
    output = args.output / "weighted_density_vs_theory.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
