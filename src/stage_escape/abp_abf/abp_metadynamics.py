import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .potential import Potential

class ABPMetaDynamics:
    def __init__(self, num_steps, td, delta_t, dimension=1, D=1.0, initial_position=None, b=Potential(1, lambda x: 0, lambda x: 0, lambda x: 0), W=0.1, sigma=0.001):
        # num_steps: Number of time steps to simulate before adding a new biasing potential

        self.num_steps = num_steps + 1
        self.td = td
        self.delta_t = delta_t
        self.dimension = dimension  # Dimension of the Brownian motion
        self.D = D  # Diffusion coefficient
        self.initial_position = initial_position if initial_position is not None else np.zeros(dimension)
        self.b = b  # Drift function
        self.b_vias = Potential(dimension, lambda x: 0, lambda x: 0, lambda x: 0)  # Initialize the biasing potential as zero
        self.W = W  # Height of the Gaussian biasing potential
        self.sigma = sigma  # Width of the Gaussian biasing potential
        self.real_time = 0.0
        self.weights = [1]  # Initialize weights for the biasing potential

        if initial_position is None:
            initial_position = np.zeros(dimension)  # Default initial position is the origin
    
        self.positions = [initial_position]  # Start at the specified initial position

    def simulate_steps(self):
        rng = np.random.default_rng()  # Use the new random number generator

        for _ in range(1, self.num_steps):
            drift = -(self.b.potential_prime_at(self.positions[-1]) + self.b_vias.potential_prime_at(self.positions[-1])) * self.delta_t  # Drift term based on the current position
            step_size = np.sqrt(2 * self.D * self.delta_t)  # Step size based on diffusion coefficient and time step
            step = step_size * rng.standard_normal(self.dimension)  # Random step from normal distribution
            new_position = self.positions[-1] + step + drift  # Add drift term
            self.real_time += np.exp(self.b_vias.potential_at(new_position) / self.D) * self.delta_t  # Update the acceleration factor based on the biasing potential
            self.positions.append(new_position)
            self.weights.append(np.exp(self.b_vias.potential_at(new_position) / self.D))  # Store the weight for the new position
        
        self.td.positions = np.array(self.positions[-self.num_steps:])  # Update the transition detector with the new positions

    
    def simulate(self, max_iters=1e6):
        for _ in range(int(max_iters)):
            self.simulate_steps()
            self.b_vias.add_gaussian(self.positions[-1], self.W, self.sigma)

            if self.td.detect_transition() is not None:
                break
        
        self.td.positions = self.td.positions
        escape_index = self.td.detect_transition()
        return self.real_time, escape_index * self.delta_t if escape_index is not None else None


    def plot_histogram_with_weights(
        self,
        ax,
        positions,
        bins=30,
        title=None,
        savepath=None,
        axis=0,
    ):
        """
        Backward-compatible wrapper for weighted histogram plotting.
        """
        from .distribution_analysis import (
            extract_coordinate,
            weighted_histogram_density,
        )
        from ..visualization import plot_histogram_density

        values = extract_coordinate(
            positions=positions,
            dimension=self.dimension,
            axis=axis,
        )

        weights = np.asarray(self.weights, dtype=float)

        if len(weights) != len(values):
            raise ValueError(
                f"weights and positions must have the same length. "
                f"Got {len(weights)} weights and {len(values)} positions."
            )

        bin_centers, density, _, _ = weighted_histogram_density(
            values=values,
            weights=weights,
            bins=bins,
        )

        plot_histogram_density(
            ax=ax,
            bin_centers=bin_centers,
            density=density,
            title=title,
        )

        if savepath is not None:
            ax.figure.savefig(savepath, dpi=300, bbox_inches="tight")

        return ax


    def plot_histogram_vs_distribution(
        self,
        positions,
        bins=30,
        dimension=None,
        axis=0,
        num_theory_points=100,
    ):
        """
        Backward-compatible wrapper for histogram vs theoretical distribution.
        """
        from .distribution_analysis import (
            extract_coordinate,
            weighted_histogram_density,
            theoretical_density_1d,
            theoretical_marginal_2d,
        )
        from ..visualization import plot_histogram_vs_distribution

        dimension = self.dimension if dimension is None else dimension

        positions = np.asarray(positions, dtype=float)
        weights = np.asarray(self.weights, dtype=float)

        values = extract_coordinate(
            positions=positions,
            dimension=dimension,
            axis=axis,
        )

        if len(weights) != len(values):
            raise ValueError(
                f"weights and positions must have the same length. "
                f"Got {len(weights)} weights and {len(values)} positions."
            )

        bin_centers, histogram_density, _, _ = weighted_histogram_density(
            values=values,
            weights=weights,
            bins=bins,
        )

        if dimension == 1:
            x_theory = np.linspace(np.min(values), np.max(values), num_theory_points)

            theoretical_density = theoretical_density_1d(
                potential=self.b,
                D=self.D,
                x_values=x_theory,
            )

        elif dimension == 2:
            if axis not in [0, 1]:
                raise ValueError("For 2D, axis must be 0 or 1.")

            x_values = np.linspace(np.min(positions[:, 0]), np.max(positions[:, 0]), num_theory_points)
            y_values = np.linspace(np.min(positions[:, 1]), np.max(positions[:, 1]), num_theory_points)

            x_theory, theoretical_density = theoretical_marginal_2d(
                potential=self.b,
                D=self.D,
                x_values=x_values,
                y_values=y_values,
                axis=axis,
            )

        else:
            raise ValueError("Only dimensions 1 and 2 are supported.")

        fig, ax = plt.subplots(figsize=(8, 5))

        plot_histogram_vs_distribution(
            ax=ax,
            bin_centers=bin_centers,
            histogram_density=histogram_density,
            x_theory=x_theory,
            theoretical_density=theoretical_density,
        )

        plt.show()

        return ax


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

            ABPMetaDynamics.set_axes_equal_3d(ax, path)

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


    def plot_final_potential(self, ax, x_range=(-2, 2), num_points=1000, title=None, savepath=None):
        """
        Plot the final potential landscape (original + biasing potential)
        on an existing matplotlib axis.
        """

        x_values = np.linspace(x_range[0], x_range[1], num_points)

        y_values = np.array([
            self.b.potential_at(np.array([x]))
            + self.b_vias.potential_at(np.array([x]))
            for x in x_values
        ])

        ax.plot(x_values, y_values, linewidth=2.0)
        ax.set_xlabel("Position")
        ax.set_ylabel("Potential")
        ax.set_title(title if title is not None else "Final Potential Landscape")
        ax.grid(True)

        if savepath is not None:
            ax.figure.savefig(savepath, dpi=300, bbox_inches="tight")

        return ax
    

    def plot_original_potential(self, ax, x_range=(-2, 2), num_points=1000, title=None, savepath=None):
        """
        Plot the original potential landscape on an existing matplotlib axis.
        """

        x_values = np.linspace(x_range[0], x_range[1], num_points)

        y_values = np.array([
            self.b.potential_at(np.array([x]))
            for x in x_values
        ])

        ax.plot(x_values, y_values, linewidth=2.0)
        ax.set_xlabel("Position")
        ax.set_ylabel("Potential")
        ax.set_title(title if title is not None else "Original Potential Landscape")
        ax.grid(True)

        if savepath is not None:
            ax.figure.savefig(savepath, dpi=300, bbox_inches="tight")

        return ax
    

    def plot_biasing_potential(self, ax, x_range=(-2, 2), num_points=1000, title=None, savepath=None):
        """
        Plot the biasing potential landscape on an existing matplotlib axis.
        """

        x_values = np.linspace(x_range[0], x_range[1], num_points)

        y_values = np.array([
            self.b_vias.potential_at(np.array([x]))
            for x in x_values
        ])

        ax.plot(x_values, y_values, linewidth=2.0)
        ax.set_xlabel("Position")
        ax.set_ylabel("Potential")
        ax.set_title(title if title is not None else "Biasing Potential Landscape")
        ax.grid(True)

        if savepath is not None:
            ax.figure.savefig(savepath, dpi=300, bbox_inches="tight")

        return ax


    def plot_all_potentials(self, x_range=(-2, 2), num_points=1000, title=None, savepath=None):
        """
        Plot the original, biasing, and final potential landscapes on a single figure.
        """

        fig, ax = plt.subplots(figsize=(8, 5))

        self.plot_original_potential(ax, x_range=x_range, num_points=num_points, title="Original Potential", savepath=None)
        self.plot_biasing_potential(ax, x_range=x_range, num_points=num_points, title="Biasing Potential", savepath=None)
        self.plot_final_potential(ax, x_range=x_range, num_points=num_points, title="Final Potential", savepath=None)

        ax.set_title(title if title is not None else "Potential Landscapes")
        ax.legend(["Original Potential", "Biasing Potential", "Final Potential"])
        ax.grid(True)

        if savepath is not None:
            plt.savefig(savepath, dpi=300, bbox_inches="tight")

        plt.show()


    def _evaluate_potential_on_grid(
        self,
        potential_type,
        x_range=(-2, 2),
        y_range=(-2, 2),
        num_points=100
    ):
        """
        Evaluate one of the potentials on a 2D grid.

        potential_type can be:
        - "original"
        - "biasing"
        - "final"
        """

        x_values = np.linspace(x_range[0], x_range[1], num_points)
        y_values = np.linspace(y_range[0], y_range[1], num_points)

        X, Y = np.meshgrid(x_values, y_values)
        Z = np.zeros_like(X, dtype=float)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.array([X[i, j], Y[i, j]])

                if potential_type == "original":
                    Z[i, j] = self.b.potential_at(point)

                elif potential_type == "biasing":
                    Z[i, j] = self.b_vias.potential_at(point)

                elif potential_type == "final":
                    Z[i, j] = (
                        self.b.potential_at(point)
                        + self.b_vias.potential_at(point)
                    )

                else:
                    raise ValueError(
                        "potential_type must be 'original', 'biasing', or 'final'"
                    )

        return X, Y, Z


    def plot_potential_contour(
        self,
        ax,
        potential_type="final",
        x_range=(-2, 2),
        y_range=(-2, 2),
        num_points=100,
        levels=20,
        title=None,
        cmap="viridis",
        add_colorbar=True
    ):
        """
        Plot level curves of a 2D potential on an existing axis.
        """

        X, Y, Z = self._evaluate_potential_on_grid(
            potential_type=potential_type,
            x_range=x_range,
            y_range=y_range,
            num_points=num_points
        )

        contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
        ax.contour(X, Y, Z, levels=levels, colors="black", linewidths=0.5)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title if title is not None else f"{potential_type.capitalize()} potential")
        ax.grid(True)

        if add_colorbar:
            ax.figure.colorbar(contour, ax=ax, shrink=0.85)

        return ax
    

    def plot_potential_surface(
        self,
        ax,
        potential_type="final",
        x_range=(-2, 2),
        y_range=(-2, 2),
        num_points=80,
        title=None,
        cmap="viridis"
    ):
        """
        Plot a 3D surface of a 2D potential on an existing 3D axis.
        """

        X, Y, Z = self._evaluate_potential_on_grid(
            potential_type=potential_type,
            x_range=x_range,
            y_range=y_range,
            num_points=num_points
        )

        surface = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="none", alpha=0.9)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("V(x, y)")
        ax.set_title(title if title is not None else f"{potential_type.capitalize()} potential")

        ax.figure.colorbar(surface, ax=ax, shrink=0.7, pad=0.1)

        return ax
    


    def plot_all_potentials_2d_3d(
        self,
        x_range=(-2, 2),
        y_range=(-2, 2),
        num_points_contour=100,
        num_points_surface=80,
        levels=20,
        savepath=None
    ):
        """
        Plot 6 graphics:
        - 3 contour plots
        - 3 3D surface plots
        """

        fig = plt.figure(figsize=(18, 10))

        ax1 = fig.add_subplot(2, 3, 1)
        self.plot_potential_contour(
            ax1,
            potential_type="original",
            x_range=x_range,
            y_range=y_range,
            num_points=num_points_contour,
            levels=levels,
            title="Original Potential - Contours"
        )

        ax2 = fig.add_subplot(2, 3, 2)
        self.plot_potential_contour(
            ax2,
            potential_type="biasing",
            x_range=x_range,
            y_range=y_range,
            num_points=num_points_contour,
            levels=levels,
            title="Biasing Potential - Contours"
        )

        ax3 = fig.add_subplot(2, 3, 3)
        self.plot_potential_contour(
            ax3,
            potential_type="final",
            x_range=x_range,
            y_range=y_range,
            num_points=num_points_contour,
            levels=levels,
            title="Final Potential - Contours"
        )

        ax4 = fig.add_subplot(2, 3, 4, projection="3d")
        self.plot_potential_surface(
            ax4,
            potential_type="original",
            x_range=x_range,
            y_range=y_range,
            num_points=num_points_surface,
            title="Original Potential - Surface"
        )

        ax5 = fig.add_subplot(2, 3, 5, projection="3d")
        self.plot_potential_surface(
            ax5,
            potential_type="biasing",
            x_range=x_range,
            y_range=y_range,
            num_points=num_points_surface,
            title="Biasing Potential - Surface"
        )

        ax6 = fig.add_subplot(2, 3, 6, projection="3d")
        self.plot_potential_surface(
            ax6,
            potential_type="final",
            x_range=x_range,
            y_range=y_range,
            num_points=num_points_surface,
            title="Final Potential - Surface"
        )

        fig.tight_layout()

        if savepath is not None:
            fig.savefig(savepath, dpi=300, bbox_inches="tight")

        plt.show()