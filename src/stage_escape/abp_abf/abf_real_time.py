import numpy as np
import matplotlib.pyplot as plt


class ABFRealTime:
    def __init__(
        self,
        num_steps,
        td,
        delta_t,
        D=1.0,
        initial_position=None,
        b=lambda x: 0,
        bins=100,
        range=(-3, 3),
        seed=None,
        rng=None,
    ):
        if seed is not None and rng is not None:
            raise ValueError("Provide either seed or rng, not both.")

        self.num_steps = num_steps
        self.td = td
        self.delta_t = delta_t
        self.bias_potential = np.zeros(bins)
        self.dimension = 1  # Dimension of the Brownian motion
        self.D = D  # Diffusion coefficient
        self.initial_position = initial_position if initial_position is not None else np.zeros(self.dimension)
        self.b = b  # Drift function
        self.force_bias = np.zeros(bins)  # Initialize force bias for each dimension and bin
        self.number_of_copies = np.zeros(bins)  # Initialize number of copies for each dimension and bin
        self.bins = bins  # Number of bins for the biasing potential
        self.range = range  # Range for the biasing potential
        self.free_energy_profile = np.zeros(bins)  # Initialize free energy profile
        self.real_time = 0.0

        if initial_position is None:
            initial_position = np.zeros(self.dimension)  # Default initial position is the origin

        self.positions = [initial_position]  # Start at the specified initial position

        self.seed = seed
        self._external_rng = rng is not None
        self.rng = (
            rng
            if rng is not None
            else np.random.default_rng(seed)
        )

    def position_to_bin(self, position):
        x = float(np.asarray(position).reshape(-1)[0])
        bin_index = int((x - self.range[0]) / (self.range[1] - self.range[0]) * self.bins)
        return np.clip(bin_index, 0, self.bins - 1)

    def _last_position(self):
        return np.asarray(self.positions[-1])

    def _current_bin(self):
        return self.position_to_bin(self.positions[-1])
    
    def _physical_force_at(self, position):
        """
        Return the derivative of the physical potential as a scalar.

        ABFRealTime is currently restricted to one dimension, while
        Potential.potential_prime_at returns a gradient vector.
        """
        gradient = np.asarray(
            self.b.potential_prime_at(
                np.asarray(position, dtype=float)
            ),
            dtype=float,
        ).reshape(-1)

        if gradient.size != self.dimension:
            raise ValueError(
                "The potential gradient must have the same dimension "
                f"as the simulation. Expected {self.dimension}, "
                f"received {gradient.size}."
            )

        return float(gradient[0])

    def simulate_steps(self):
        for _ in range(1, self.num_steps):
            last_position = self._last_position()
            x = float(last_position.reshape(-1)[0])

            if x < self.range[0] or x > self.range[1]:
                raise ValueError(
                    f"Position {last_position} is outside the configured "
                    f"range {self.range}."
                )

            current_bin = self._current_bin()

            physical_force = self._physical_force_at(
                last_position
            )

            drift = (
                -physical_force
                + self.force_bias[current_bin]
            ) * self.delta_t

            step_size = np.sqrt(
                2.0 * self.D * self.delta_t
            )

            step = (
                step_size
                * self.rng.standard_normal(self.dimension)
            )

            new_position = (
                last_position
                + step
                + np.array([drift], dtype=float)
            )

            self.real_time += (
                np.exp(
                    self.bias_potential[current_bin] / self.D
                )
                * self.delta_t
            )

            self.positions.append(new_position)

        self.td.positions = np.asarray(
            self.positions[-self.num_steps:],
            dtype=float,
        )

    def simulate(self, max_iters=1e6):
        while len(self.positions) < max_iters:
            self.simulate_steps()

            current_bin = self._current_bin()
            n = self.number_of_copies[current_bin]

            physical_force = self._physical_force_at(
                self.positions[-1]
            )

            self.force_bias[current_bin] = (
                n * self.force_bias[current_bin]
                + physical_force
            ) / (n + 1)

            self.number_of_copies[current_bin] += 1

            self.free_energy()

            escape_index = self.td.detect_transition()

            if escape_index is not None:
                break

        return (
            self.real_time,
            (
                escape_index * self.delta_t
                if escape_index is not None
                else None
            ),
        )

    def free_energy(self):
        # Calculate the free energy profile based on the force bias and number of copies
        free_energy = np.zeros(self.bins)
        d_x = (self.range[1] - self.range[0]) / self.bins  # Width of each bin
        free_energy[0] = -self.force_bias[0] * d_x
        for bin_index in range(1, self.bins):
            free_energy[bin_index] = free_energy[bin_index - 1] - self.force_bias[bin_index] * d_x

        self.free_energy_profile = free_energy
        return self.free_energy_profile

    def update_bias_potential(self):
        dx = (self.range[1] - self.range[0]) / self.bins

        # A'(x) ≈ force_bias
        A = np.zeros(self.bins)

        # Integración trapezoidal más estable que rectángulos simples
        A[1:] = np.cumsum(
            0.5 * (self.force_bias[:-1] + self.force_bias[1:]) * dx
        )

        # Potencial de sesgo: V_bias = -A + C
        V_bias = -A

        # Gauge: hacer que el sesgo sea >= 0, como en ABP/metadynamics
        # Esto importa para exp(V_bias / D)
        V_bias -= np.min(V_bias)

        self.bias_potential = V_bias
        return self.bias_potential

    def plot_free_energy(self):
        bin_centers = np.linspace(self.range[0], self.range[1], self.bins)  # Calculate bin centers for plotting
        plt.plot(bin_centers, self.free_energy_profile)  # Plot the free energy profile
        plt.xlabel('Position')  # Label for x-axis
        plt.ylabel('Free Energy')  # Label for y-axis
        plt.title('Free Energy Profile')  # Title of the plot
        plt.grid()  # Add grid to the plot
        plt.show()  # Display the plot