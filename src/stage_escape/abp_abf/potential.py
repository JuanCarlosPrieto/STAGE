import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class Potential:
    def __init__(self, dimension, function, first_derivative=None, second_derivative=None):
        self.dimension = dimension
        self.potential = function
        self.potential_prime = first_derivative
        self.potential_biprime = second_derivative

    
    def potential_at(self, point):
        if point.shape[-1] != self.dimension:
            raise ValueError("Function dimension does not coincide with point dimension")
        
        return self.potential(point)
    
    
    def potential_prime_at(self, point, epsilon=1e-3):
        if self.potential_prime is None:
            gradient = np.zeros(self.dimension)
            for i in range(self.dimension):
                h = np.zeros(self.dimension)
                h[i] = epsilon
                gradient[i] = (self.potential_at(point + h) - self.potential_at(point)) / epsilon
            
            return gradient
        
        return self.potential_prime(point)


    def potential_biprime_at(self, point, epsilon=1e-3):
        if self.potential_biprime is None:
            hessian = np.zeros((self.dimension, self.dimension))
            for i in range(self.dimension):
                for j in range(self.dimension):
                    h_i = np.zeros(self.dimension)
                    h_j = np.zeros(self.dimension)
                    h_i[i] = epsilon
                    h_j[j] = epsilon
                    f_ij = self.potential_at(point + h_i + h_j)
                    f_i = self.potential_at(point + h_i)
                    f_j = self.potential_at(point + h_j)
                    f_0 = self.potential_at(point)
                    hessian[i, j] = (f_ij - f_i - f_j + f_0) / (epsilon ** 2)
            
            return hessian
        
        return self.potential_biprime(point)
    

    def add_gaussian(self, center, height, width):
        def gaussian(x):
            diff = x - center
            exponent = -np.sum(diff ** 2) / (2 * width ** 2)
            return height * np.exp(exponent)

        def gaussian_prime(x):
            diff = x - center
            exponent = -np.sum(diff ** 2) / (2 * width ** 2)
            return -height * np.exp(exponent) * diff / (width ** 2)

        def gaussian_biprime(x):
            diff = x - center
            exponent = -np.sum(diff ** 2) / (2 * width ** 2)
            outer_product = np.outer(diff, diff)
            return height * np.exp(exponent) * ((outer_product / (width ** 4)) - (np.eye(self.dimension) / (width ** 2)))

        # Update the potential and its derivatives
        old_potential = self.potential
        old_potential_prime = self.potential_prime
        old_potential_biprime = self.potential_biprime

        self.potential = lambda x: old_potential(x) + gaussian(x)
        self.potential_prime = lambda x: old_potential_prime(x) + gaussian_prime(x) if old_potential_prime is not None else gaussian_prime(x)
        self.potential_biprime = lambda x: old_potential_biprime(x) + gaussian_biprime(x) if old_potential_biprime is not None else gaussian_biprime(x)


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