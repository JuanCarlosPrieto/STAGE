# visualization.py

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize


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
    """
    Add a 2D path to an existing matplotlib axis.
    The path is colored from red to violet according to progression.
    """

    path = np.asarray(path, dtype=float)

    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must have shape (N, 2).")

    n_steps = len(path)

    if n_steps < 2:
        raise ValueError("path must contain at least two points.")

    cmap = LinearSegmentedColormap.from_list(
        "red_to_violet",
        ["red", "violet"]
    )

    norm_segments = Normalize(vmin=0, vmax=n_steps - 2)
    segment_colors = cmap(norm_segments(np.arange(n_steps - 1)))

    points = path[:, :2]
    segments = np.stack([points[:-1], points[1:]], axis=1)

    line_collection = LineCollection(
        segments,
        colors=segment_colors,
        linewidths=linewidth,
        label=label,
    )

    ax.add_collection(line_collection)

    if show_points:
        point_indices = np.arange(0, n_steps, point_stride)

        # Make sure the last point is included among the plotted points.
        if point_indices[-1] != n_steps - 1:
            point_indices = np.append(point_indices, n_steps - 1)

        norm_points = Normalize(vmin=0, vmax=n_steps - 1)
        point_colors = cmap(norm_points(point_indices))

        ax.scatter(
            path[point_indices, 0],
            path[point_indices, 1],
            c=point_colors,
            s=point_size,
            alpha=point_alpha,
            edgecolors="none",
        )

    ax.scatter(path[0, 0], path[0, 1], color="red", s=60, label="Start")
    ax.scatter(path[-1, 0], path[-1, 1], color="violet", s=60, label="End")

    return line_collection, cmap, norm_segments