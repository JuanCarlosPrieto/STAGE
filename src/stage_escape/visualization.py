from __future__ import annotations

import numpy as np


def _validate_stride(point_stride: int) -> int:
    if isinstance(point_stride, bool) or not isinstance(
        point_stride, (int, np.integer)
    ):
        raise TypeError("point_stride must be an integer.")
    point_stride = int(point_stride)
    if point_stride < 1:
        raise ValueError("point_stride must be at least 1.")
    return point_stride


def _progress_colormap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("red_to_violet", ["red", "violet"])


def add_colored_path_2d(
    ax,
    path,
    point_stride=1000,
    point_size=50,
    point_alpha=0.8,
    show_points=True,
    linewidth=2.0,
    label="Brownian trajectory",
):
    """Add a progression-colored 2D path to an existing axis."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must have shape (N, 2).")
    if len(path) < 2:
        raise ValueError("path must contain at least two points.")
    if not np.all(np.isfinite(path)):
        raise ValueError("path must contain finite values.")
    point_stride = _validate_stride(point_stride)

    n_steps = len(path)
    cmap = _progress_colormap()
    norm_segments = Normalize(vmin=0, vmax=max(n_steps - 2, 1))
    colors = cmap(norm_segments(np.arange(n_steps - 1)))
    segments = np.stack([path[:-1], path[1:]], axis=1)
    line_collection = LineCollection(
        segments,
        colors=colors,
        linewidths=linewidth,
        label=label,
    )
    ax.add_collection(line_collection)

    if show_points:
        point_indices = np.arange(0, n_steps, point_stride)
        if point_indices[-1] != n_steps - 1:
            point_indices = np.append(point_indices, n_steps - 1)
        point_norm = Normalize(vmin=0, vmax=max(n_steps - 1, 1))
        ax.scatter(
            path[point_indices, 0],
            path[point_indices, 1],
            c=cmap(point_norm(point_indices)),
            s=point_size,
            alpha=point_alpha,
            edgecolors="none",
        )
    ax.scatter(path[0, 0], path[0, 1], color="red", s=60, label="Start")
    ax.scatter(path[-1, 0], path[-1, 1], color="violet", s=60, label="End")
    return line_collection, cmap, norm_segments


def set_axes_equal_3d(ax, points):
    """Force equal data scale on a 3D axis."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points must have shape (N, 3).")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain finite values.")
    minima = points.min(axis=0)
    maxima = points.max(axis=0)
    middle = 0.5 * (minima + maxima)
    radius = 0.5 * float(np.max(maxima - minima))
    if radius == 0.0:
        radius = 0.5
    ax.set_xlim(middle[0] - radius, middle[0] + radius)
    ax.set_ylim(middle[1] - radius, middle[1] + radius)
    ax.set_zlim(middle[2] - radius, middle[2] + radius)
    return ax


def plot_brownian_path(
    path,
    title=None,
    mode_1d="space",
    savepath=None,
    show_points=True,
    point_size=50,
    point_alpha=0.8,
    point_stride=1,
    show=False,
):
    """Plot a Brownian path and return ``(fig, ax)``.

    The function no longer calls ``plt.show`` unless ``show=True``. This makes
    it safe for scripts, tests and report-generation pipelines.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    path = np.asarray(path, dtype=float)
    if path.ndim == 1:
        path = path.reshape(-1, 1)
    if path.ndim != 2:
        raise ValueError("path must have shape (N,), (N, 1), (N, 2), or (N, 3).")
    if not np.all(np.isfinite(path)):
        raise ValueError("path must contain finite values.")
    n_steps, dimension = path.shape
    if dimension not in {1, 2, 3}:
        raise ValueError("Only 1D, 2D and 3D paths are supported.")
    if n_steps < 2:
        raise ValueError("The path must contain at least two points.")
    point_stride = _validate_stride(point_stride)

    cmap = _progress_colormap()
    norm = Normalize(vmin=0, vmax=max(n_steps - 2, 1))
    segment_colors = cmap(norm(np.arange(n_steps - 1)))
    point_indices = np.arange(0, n_steps, point_stride)
    if point_indices[-1] != n_steps - 1:
        point_indices = np.append(point_indices, n_steps - 1)
    point_norm = Normalize(vmin=0, vmax=max(n_steps - 1, 1))
    point_colors = cmap(point_norm(point_indices))

    if dimension == 1:
        x = path[:, 0]
        if mode_1d == "space":
            fig, ax = plt.subplots(figsize=(8, 2.5))
            y = np.zeros_like(x)
            points = np.column_stack([x, y])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            ax.add_collection(
                LineCollection(segments, colors=segment_colors, linewidths=2.5)
            )
            if show_points:
                ax.scatter(
                    x[point_indices],
                    y[point_indices],
                    c=point_colors,
                    s=point_size,
                    alpha=point_alpha,
                    edgecolors="none",
                )
            ax.scatter(x[0], 0, color="red", s=50, label="Start")
            ax.scatter(x[-1], 0, color="violet", s=50, label="End")
            spread = float(np.ptp(x))
            margin = 0.05 * spread if spread > 0.0 else 0.5
            ax.set_xlim(x.min() - margin, x.max() + margin)
            ax.set_ylim(-0.1, 0.1)
            ax.set_xlabel("Position")
            ax.set_yticks([])
            ax.legend()
        elif mode_1d == "time_series":
            fig, ax = plt.subplots(figsize=(8, 4))
            time = np.arange(n_steps)
            points = np.column_stack([time, x])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            ax.add_collection(
                LineCollection(segments, colors=segment_colors, linewidths=2.5)
            )
            if show_points:
                ax.scatter(
                    time[point_indices],
                    x[point_indices],
                    c=point_colors,
                    s=point_size,
                    alpha=point_alpha,
                    edgecolors="none",
                )
            ax.scatter(time[0], x[0], color="red", s=50, label="Start")
            ax.scatter(time[-1], x[-1], color="violet", s=50, label="End")
            ax.set_xlim(time.min(), time.max())
            spread = float(np.ptp(x))
            margin = 0.05 * spread if spread > 0.0 else 0.5
            ax.set_ylim(x.min() - margin, x.max() + margin)
            ax.set_xlabel("Step")
            ax.set_ylabel("X")
            ax.legend()
        else:
            raise ValueError("mode_1d must be 'space' or 'time_series'.")
    elif dimension == 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        add_colored_path_2d(
            ax,
            path,
            point_stride=point_stride,
            point_size=point_size,
            point_alpha=point_alpha,
            show_points=show_points,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")
        ax.autoscale()
        ax.legend()
    else:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        segments = np.stack([path[:-1], path[1:]], axis=1)
        ax.add_collection3d(
            Line3DCollection(segments, colors=segment_colors, linewidths=2.0)
        )
        if show_points:
            ax.scatter(
                path[point_indices, 0],
                path[point_indices, 1],
                path[point_indices, 2],
                c=point_colors,
                s=point_size,
                alpha=point_alpha,
                edgecolors="none",
            )
        ax.scatter(*path[0], color="red", s=50, label="Start")
        ax.scatter(*path[-1], color="violet", s=50, label="End")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_axes_equal_3d(ax, path)
        ax.legend()

    if title is not None:
        ax.set_title(title)
    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.03, pad=0.04)
    colorbar.set_label("Trajectory progression")
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_narrow_escape_problem(
    *,
    path,
    surface,
    escapes,
    escape_checker,
    xlim,
    ylim,
    point_stride=1000,
    point_size=50,
    point_alpha=0.8,
    show_points=True,
    show=False,
):
    import matplotlib.pyplot as plt

    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("This plotting function requires a path of shape (N, 2).")
    fig, ax = plt.subplots(figsize=(7, 7))
    _, cmap, norm = add_colored_path_2d(
        ax=ax,
        path=path,
        point_stride=point_stride,
        point_size=point_size,
        point_alpha=point_alpha,
        show_points=show_points,
    )
    surface.plot_boundary_2d(
        ax=ax,
        xlim=xlim,
        ylim=ylim,
        escape_checker=escape_checker,
    )
    if escape_checker is None:
        for escape in escapes:
            escape.plot_escape(ax=ax, xlim=xlim, ylim=ylim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal")
    ax.autoscale()
    ax.legend()
    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.03, pad=0.04)
    colorbar.set_label("Trajectory progression")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_histogram_density(
    ax,
    bin_centers,
    density,
    title=None,
    label="Weighted histogram",
    xlabel="Position",
    ylabel="Weighted density",
    show_curve=True,
    fill=True,
    alpha=0.4,
):
    """Plot a density whose equally spaced bin centers are already known."""
    bin_centers = np.asarray(bin_centers, dtype=float)
    density = np.asarray(density, dtype=float)
    if bin_centers.ndim != 1 or density.ndim != 1:
        raise ValueError("bin_centers and density must be one-dimensional.")
    if len(bin_centers) != len(density):
        raise ValueError("bin_centers and density must have the same length.")
    if len(bin_centers) < 2:
        raise ValueError("At least two bin centers are required.")
    differences = np.diff(bin_centers)
    width = differences[0]
    if width <= 0.0 or not np.allclose(differences, width):
        raise ValueError("bin_centers must be equally spaced and increasing.")
    edges = np.concatenate(
        ([bin_centers[0] - width / 2.0], bin_centers + width / 2.0)
    )
    ax.stairs(density, edges, fill=fill, alpha=alpha, label=label)
    if show_curve:
        ax.plot(
            bin_centers,
            density,
            marker="o",
            linestyle="-",
            label=f"{label} centers",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Weighted histogram")
    ax.grid(True)
    ax.legend()
    return ax


def plot_histogram_vs_distribution(
    ax,
    bin_centers,
    histogram_density,
    x_theory,
    theoretical_density,
    title=None,
):
    ax.plot(
        bin_centers,
        histogram_density,
        marker="o",
        linestyle="-",
        label="Weighted histogram",
    )
    ax.plot(x_theory, theoretical_density, linewidth=2.0, label="Theory")
    ax.set_xlabel("Position")
    ax.set_ylabel("Density")
    ax.set_title(title or "Histogram vs theoretical distribution")
    ax.grid(True)
    ax.legend()
    return ax
