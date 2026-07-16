import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

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


def plot_final_potential(self, ax, x_range=(-2, 2), num_points=1000, title=None, savepath=None):
    """
    Plot the final potential landscape (original + biasing potential)
    on an existing matplotlib axis.
    """

    x_values = np.linspace(x_range[0], x_range[1], num_points)

    y_values = np.array([
        self.b.potential_at(np.array([x]))
        + self.bias_potential_at(np.array([x]))
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
    

    
    

def plot_biasing_potential(self, ax, x_range=(-2, 2), num_points=1000, title=None, savepath=None):
    """
    Plot the biasing potential landscape on an existing matplotlib axis.
    """

    x_values = np.linspace(x_range[0], x_range[1], num_points)

    y_values = np.array([
        self.bias_potential_at(np.array([x]))
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


def plot_potential(
    self,
    ax=None,
    x_range=(-2, 2),
    y_range=(-2, 2),
    num_points=250,
    levels=60,
    contour_levels=20,
    filled=True,
    cmap="plasma",
    contrast_gamma=None,
    clip_percentiles=(1, 99),
    show_colorbar=True,
    show_contour_lines=True,
    contour_color="black",
    contour_linewidths=0.7,
    contour_alpha=0.7,
    show_labels=False,
    show_gradient=False,
    gradient_direction="minus_gradient",
    gradient_mode="finite_difference",
    gradient_epsilon=1e-4,
    arrow_stride=12,
    normalize_arrows=True,
    arrow_scale=30,
    arrow_width=0.003,
    arrow_color="white",
    arrow_alpha=0.85,
    title=None,
    savepath=None
):
    """
    Plot the potential.

    If dimension == 1:
        plots V(x).

    If dimension == 2:
        plots V(x, y) using contourf / contour lines.
        Optionally overlays the vector field given by grad(V) or -grad(V).

    Parameters
    ----------
    show_gradient : bool
        If True, overlays arrows corresponding to the gradient field.

    gradient_direction : str
        "gradient"       -> arrows point in the direction of ∇V
        "minus_gradient" -> arrows point in the direction of -∇V

    gradient_mode : str
        "finite_difference" -> computes gradient numerically from V
        "analytic"          -> uses self.potential_prime_at

    normalize_arrows : bool
        If True, all arrows have similar length and only show direction.
        If False, arrow length is proportional to gradient magnitude.

    clip_percentiles : tuple or None
        Clips the plotted potential values between percentiles.
        This makes transitions visually stronger if V has large extreme values.

    contrast_gamma : float or None
        If not None, applies PowerNorm contrast.
        Values < 1 usually enhance low-value regions.
    """

    if self.dimension not in [1, 2]:
        raise ValueError("plot_potential only supports dimension 1 or 2")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    def scalar_potential(point):
        value = np.asarray(self.potential_at(point), dtype=float)

        if value.size != 1:
            raise ValueError(
                "The potential function must return a scalar when evaluated at one point."
            )

        return float(value.reshape(-1)[0])

    def finite_difference_gradient(point):
        grad = np.zeros(self.dimension)

        for i in range(self.dimension):
            h = np.zeros(self.dimension)
            h[i] = gradient_epsilon

            f_plus = scalar_potential(point + h)
            f_minus = scalar_potential(point - h)

            grad[i] = (f_plus - f_minus) / (2 * gradient_epsilon)

        return grad

    def gradient_for_plot(point):
        if gradient_mode == "finite_difference":
            return finite_difference_gradient(point)

        elif gradient_mode == "analytic":
            grad = np.asarray(self.potential_prime_at(point), dtype=float).reshape(-1)

            if grad.size != self.dimension:
                raise ValueError(
                    "potential_prime_at must return a vector with the same dimension as the potential."
                )

            return grad

        else:
            raise ValueError("gradient_mode must be 'finite_difference' or 'analytic'")

    if self.dimension == 1:
        x_values = np.linspace(x_range[0], x_range[1], num_points)

        V_values = np.array([
            scalar_potential(np.array([x]))
            for x in x_values
        ])

        ax.plot(x_values, V_values, linewidth=2.0)
        ax.set_xlabel("x")
        ax.set_ylabel("V(x)")
        ax.set_title(title if title is not None else "Potential V(x)")
        ax.grid(True)

    elif self.dimension == 2:
        x_values = np.linspace(x_range[0], x_range[1], num_points)
        y_values = np.linspace(y_range[0], y_range[1], num_points)

        X, Y = np.meshgrid(x_values, y_values)

        points = np.column_stack([X.ravel(), Y.ravel()])

        Z = np.array([
            scalar_potential(point)
            for point in points
        ]).reshape(X.shape)

        finite_values = Z[np.isfinite(Z)]

        if finite_values.size == 0:
            raise ValueError("The potential only produced non-finite values.")

        if clip_percentiles is not None:
            vmin, vmax = np.percentile(finite_values, clip_percentiles)
            Z_plot = np.clip(Z, vmin, vmax)
        else:
            vmin, vmax = np.min(finite_values), np.max(finite_values)
            Z_plot = Z

        if contrast_gamma is not None:
            norm = colors.PowerNorm(
                gamma=contrast_gamma,
                vmin=vmin,
                vmax=vmax
            )
        else:
            norm = None

        if filled:
            contour_filled = ax.contourf(
                X,
                Y,
                Z_plot,
                levels=levels,
                cmap=cmap,
                norm=norm
            )

            if show_colorbar:
                fig.colorbar(contour_filled, ax=ax, label="V(x, y)")

        if show_contour_lines:
            contour_lines = ax.contour(
                X,
                Y,
                Z_plot,
                levels=contour_levels,
                colors=contour_color,
                linewidths=contour_linewidths,
                alpha=contour_alpha
            )

            if show_labels:
                ax.clabel(contour_lines, inline=True, fontsize=8)

        if show_gradient:
            Xq = X[::arrow_stride, ::arrow_stride]
            Yq = Y[::arrow_stride, ::arrow_stride]

            points_q = np.column_stack([Xq.ravel(), Yq.ravel()])

            gradients = np.array([
                gradient_for_plot(point)
                for point in points_q
            ])

            U = gradients[:, 0].reshape(Xq.shape)
            V = gradients[:, 1].reshape(Yq.shape)

            if gradient_direction == "gradient":
                vector_label = r"$\nabla V$"

            elif gradient_direction == "minus_gradient":
                U = -U
                V = -V
                vector_label = r"$-\nabla V$"

            else:
                raise ValueError(
                    "gradient_direction must be 'gradient' or 'minus_gradient'"
                )

            if normalize_arrows:
                magnitude = np.sqrt(U**2 + V**2)
                magnitude[magnitude == 0] = 1.0

                U = U / magnitude
                V = V / magnitude

            ax.quiver(
                Xq,
                Yq,
                U,
                V,
                color=arrow_color,
                alpha=arrow_alpha,
                scale=arrow_scale,
                width=arrow_width,
                label=vector_label
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title if title is not None else "Potential V(x, y)")
        ax.set_aspect("equal", adjustable="box")

        if show_gradient:
            ax.legend(loc="upper right")

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return ax