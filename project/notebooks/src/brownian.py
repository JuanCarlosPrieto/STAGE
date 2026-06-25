# Brownian Motion Simulation

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class BrownianMotion:
    def __init__(self, num_steps, delta_t, dimension=1, D=1, initial_position=None, b=lambda x: 0):
        self.num_steps = num_steps
        self.delta_t = delta_t
        self.dimension = dimension  # Dimension of the Brownian motion
        self.D = D  # Diffusion coefficient
        self.b = b  # Drift term

        if initial_position is None:
            initial_position = np.zeros(dimension)  # Default initial position is the origin
    
        self.positions = [initial_position]  # Start at the specified initial position        


    def simulate(self):
        rng = np.random.default_rng()  # Use the new random number generator

        for _ in range(1, self.num_steps):
            drift = self.b(self.positions[-1]) * self.delta_t  # Drift term based on the current position
            step_size = np.sqrt(2 * self.D * self.delta_t)  # Step size based on diffusion coefficient and time step
            step = step_size * rng.standard_normal(self.dimension)  # Random step from normal distribution
            new_position = self.positions[-1] + step + drift  # Add drift term
            self.positions.append(new_position)
        return np.array(self.positions)
    

    @staticmethod
    def plot_brownian_path(
        path,
        title=None,
        mode_1d="space",
        savepath=None,
        show_points=True,
        point_size=50,
        point_alpha=0.8,
        point_stride=1
    ):
        """
        Plot a Brownian trajectory in 1D, 2D or 3D.
        
        Parameters
        ----------
        path : array-like
            Brownian path.
            Shape can be:
                (N,)      for 1D
                (N, 1)    for 1D
                (N, 2)    for 2D
                (N, 3)    for 3D

        title : str, optional
            Figure title.

        mode_1d : str
            Only used for 1D paths.
            "space"       -> plot the trajectory on a line, with no explicit time axis.
            "time_series" -> plot X(t) versus step index.

        savepath : str, optional
            If provided, saves the figure to this path.
        """

        path = np.asarray(path, dtype=float)

        if path.ndim == 1:
            path = path.reshape(-1, 1)

        if path.ndim != 2:
            raise ValueError("path must have shape (N,), (N, 1), (N, 2), or (N, 3).")

        n_steps, dim = path.shape

        if dim not in [1, 2, 3]:
            raise ValueError("Only 1D, 2D and 3D paths are supported.")

        if n_steps < 2:
            raise ValueError("The path must contain at least two points.")

        cmap = LinearSegmentedColormap.from_list(
            "red_to_violet",
            ["red", "violet"]
        )

        norm = Normalize(vmin=0, vmax=n_steps - 2)
        colors = cmap(norm(np.arange(n_steps - 1)))

        point_indices = np.arange(0, n_steps, point_stride)
        point_norm = Normalize(vmin=0, vmax=n_steps - 1)
        point_colors = cmap(point_norm(point_indices))

        if dim == 1:
            x = path[:, 0]

            if mode_1d == "space":
                fig, ax = plt.subplots(figsize=(8, 2.5))

                y = np.zeros_like(x)

                points = np.column_stack([x, y])
                segments = np.stack([points[:-1], points[1:]], axis=1)

                line_collection = LineCollection(
                    segments,
                    colors=colors,
                    linewidths=2.5
                )

                ax.add_collection(line_collection)

                if show_points:
                    ax.scatter(
                        x[point_indices],
                        y[point_indices],
                        c=point_colors,
                        s=point_size,
                        alpha=point_alpha,
                        edgecolors="none"
                    )

                ax.scatter(x[0], 0, color="red", s=50, label="Start")
                ax.scatter(x[-1], 0, color="violet", s=50, label="End")

                margin = 0.05 * (x.max() - x.min() + 1e-12)
                ax.set_xlim(x.min() - margin, x.max() + margin)
                ax.set_ylim(-0.1, 0.1)

                ax.set_xlabel("Position")
                ax.set_yticks([])
                ax.set_ylabel("")
                ax.legend()

            elif mode_1d == "time_series":
                fig, ax = plt.subplots(figsize=(8, 4))

                t = np.arange(n_steps)
                points = np.column_stack([t, x])
                segments = np.stack([points[:-1], points[1:]], axis=1)

                line_collection = LineCollection(
                    segments,
                    colors=colors,
                    linewidths=2.5
                )

                ax.add_collection(line_collection)

                if show_points:
                    ax.scatter(
                        t[point_indices],
                        x[point_indices],
                        c=point_colors,
                        s=point_size,
                        alpha=point_alpha,
                        edgecolors="none"
                    )

                ax.scatter(t[0], x[0], color="red", s=50, label="Start")
                ax.scatter(t[-1], x[-1], color="violet", s=50, label="End")

                ax.set_xlim(t.min(), t.max())
                ax.set_ylim(x.min(), x.max())

                ax.set_xlabel("Step")
                ax.set_ylabel("X")
                ax.legend()

            else:
                raise ValueError("mode_1d must be 'space' or 'time_series'.")

        elif dim == 2:
            fig, ax = plt.subplots(figsize=(6, 6))

            points = path[:, :2]
            segments = np.stack([points[:-1], points[1:]], axis=1)

            line_collection = LineCollection(
                segments,
                colors=colors,
                linewidths=2.0
            )

            ax.add_collection(line_collection)

            if show_points:
                ax.scatter(
                    path[point_indices, 0],
                    path[point_indices, 1],
                    c=point_colors,
                    s=point_size,
                    alpha=point_alpha,
                    edgecolors="none"
                )

            ax.scatter(path[0, 0], path[0, 1], color="red", s=50, label="Start")
            ax.scatter(path[-1, 0], path[-1, 1], color="violet", s=50, label="End")

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.axis("equal")
            ax.autoscale()
            ax.legend()

        else:
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")

            points = path[:, :3]
            segments = np.stack([points[:-1], points[1:]], axis=1)

            line_collection = Line3DCollection(
                segments,
                colors=colors,
                linewidths=2.0
            )

            ax.add_collection3d(line_collection)

            if show_points:
                ax.scatter(
                    path[point_indices, 0],
                    path[point_indices, 1],
                    path[point_indices, 2],
                    c=point_colors,
                    s=point_size,
                    alpha=point_alpha,
                    edgecolors="none"
                )

            ax.scatter(path[0, 0], path[0, 1], path[0, 2], color="red", s=50, label="Start")
            ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], color="violet", s=50, label="End")

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            BrownianMotion.set_axes_equal_3d(ax, path)

            ax.legend()

        if title is not None:
            ax.set_title(title)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Trajectory progression")

        plt.tight_layout()

        if savepath is not None:
            plt.savefig(savepath, dpi=300, bbox_inches="tight")

        plt.show()



    @staticmethod
    def set_axes_equal_3d(ax, points):
        """
        Force equal scale on a 3D matplotlib plot.
        """

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        x_middle = 0.5 * (x.max() + x.min())
        y_middle = 0.5 * (y.max() + y.min())
        z_middle = 0.5 * (z.max() + z.min())

        radius = 0.5 * max(
            x.max() - x.min(),
            y.max() - y.min(),
            z.max() - z.min()
        )

        ax.set_xlim(x_middle - radius, x_middle + radius)
        ax.set_ylim(y_middle - radius, y_middle + radius)
        ax.set_zlim(z_middle - radius, z_middle + radius)