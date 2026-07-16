from __future__ import annotations

import numpy as np


def _finite_1d(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


class Statistics:
    """Numerical estimators and opt-in plotting helpers."""

    @staticmethod
    def adapt_exp_distribution(data, t_min=0.0) -> float:
        data = _finite_1d(data, name="data")
        t_min = float(t_min)
        if not np.isfinite(t_min):
            raise ValueError("t_min must be finite.")
        filtered = data[data >= t_min]
        if filtered.size == 0:
            raise ValueError("No data points are greater than or equal to t_min.")
        shifted_mean = float(np.mean(filtered) - t_min)
        if shifted_mean <= 0.0:
            raise ValueError("The shifted sample mean must be strictly positive.")
        return 1.0 / shifted_mean

    @staticmethod
    def plot_mean_times(
        x,
        mean_times,
        std_devs,
        xlabel="X-axis",
        ylabel="Mean Times",
        title="Mean Times with Standard Deviation",
        *,
        ax=None,
        show=False,
    ):
        import matplotlib.pyplot as plt

        x = _finite_1d(x, name="x")
        mean_times = _finite_1d(mean_times, name="mean_times")
        std_devs = _finite_1d(std_devs, name="std_devs")
        if not (len(x) == len(mean_times) == len(std_devs)):
            raise ValueError("x, mean_times and std_devs must have equal lengths.")
        if np.any(std_devs < 0.0):
            raise ValueError("std_devs must be non-negative.")
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(x, mean_times, yerr=std_devs, fmt="-o", capsize=5)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid(True)
        if show:
            plt.show()
        return ax.figure, ax

    @staticmethod
    def plot_histogram(
        data,
        bins=30,
        xlabel="Data",
        ylabel="Frequency",
        title="Histogram",
        *,
        density=False,
        ax=None,
        show=False,
    ):
        import matplotlib.pyplot as plt

        data = _finite_1d(data, name="data")
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data, bins=bins, density=density, alpha=0.7, edgecolor="black")
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid(True)
        if show:
            plt.show()
        return ax.figure, ax

    @staticmethod
    def plot_histogram_with_fit(
        data,
        bins=30,
        xlabel="Data",
        ylabel="Density",
        title="Histogram with Exponential Fit",
        t_min=0.0,
        *,
        ax=None,
        show=False,
    ):
        import matplotlib.pyplot as plt

        data = _finite_1d(data, name="data")
        rate = Statistics.adapt_exp_distribution(data, t_min=t_min)
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data, bins=bins, alpha=0.7, edgecolor="black", density=True)
        x = np.linspace(float(np.min(data)), float(np.max(data)), 200)
        pdf = np.where(x >= t_min, rate * np.exp(-rate * (x - t_min)), 0.0)
        ax.plot(x, pdf, linewidth=2, label=f"Exponential fit (λ={rate:.3g})")
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid(True)
        ax.legend()
        if show:
            plt.show()
        return ax.figure, ax

    @staticmethod
    def plot_graph_and_linear_adjustment(
        x,
        y,
        xlabel="X-axis",
        ylabel="Y-axis",
        title="Graph with Linear Adjustment",
        *,
        ax=None,
        show=False,
    ):
        import matplotlib.pyplot as plt

        x = _finite_1d(x, name="x")
        y = _finite_1d(y, name="y")
        if len(x) != len(y) or len(x) < 2:
            raise ValueError("x and y must have equal lengths of at least two.")
        coefficients = np.polyfit(x, y, 1)
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y, label="Data")
        ax.plot(
            x,
            np.polyval(coefficients, x),
            label=f"Linear fit: y={coefficients[0]:.3g}x+{coefficients[1]:.3g}",
        )
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid(True)
        ax.legend()
        if show:
            plt.show()
        return ax.figure, ax, coefficients

    @staticmethod
    def simple_plot(
        x,
        y,
        xlabel="X-axis",
        ylabel="Y-axis",
        title="Simple Plot",
        *,
        ax=None,
        show=False,
    ):
        import matplotlib.pyplot as plt

        x = _finite_1d(x, name="x")
        y = _finite_1d(y, name="y")
        if len(x) != len(y):
            raise ValueError("x and y must have equal lengths.")
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, marker="o")
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid(True)
        if show:
            plt.show()
        return ax.figure, ax
