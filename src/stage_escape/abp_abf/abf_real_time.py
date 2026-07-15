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
    
    def _potential_gradient_scalar(self, position) -> float:
        """Return the scalar derivative of a one-dimensional potential."""
        gradient = np.asarray(
            self.b.potential_prime_at(position),
            dtype=float,
        ).reshape(-1)

        if gradient.size != 1:
            raise ValueError(
                "ABFRealTime only supports one-dimensional potentials. "
                f"Received {gradient.size} gradient components."
            )

        return float(gradient[0])
    
    def _update_force_estimator(self, position) -> int:
        """
        Update the running mean-force estimator using one trajectory sample.

        Returns
        -------
        int
            Index of the bin updated by this sample.
        """
        current_bin = self.position_to_bin(position)
        sample_force = self._potential_gradient_scalar(position)

        n = self.number_of_copies[current_bin]

        self.force_bias[current_bin] = (
            n * self.force_bias[current_bin]
            + sample_force
        ) / (n + 1)

        self.number_of_copies[current_bin] = n + 1

        return current_bin
    
    def simulate_one_step(self) -> bool:
        """
        Perform one ABF integration step.

        Returns
        -------
        bool
            True when a transition is detected after the step.
        """
        last_position = self._last_position()
        x = float(last_position.reshape(-1)[0])

        if x < self.range[0] or x > self.range[1]:
            raise ValueError(
                "Position out of bounds. "
                f"Received x={x}, expected a value inside {self.range}."
            )

        current_bin = self.position_to_bin(last_position)
        physical_gradient = self._potential_gradient_scalar(
            last_position
        )

        drift = np.array(
            [
                (
                    -physical_gradient
                    + self.force_bias[current_bin]
                )
                * self.delta_t
            ],
            dtype=float,
        )

        noise = (
            np.sqrt(2.0 * self.D * self.delta_t)
            * self.rng.standard_normal(self.dimension)
        )

        # The physical-time increment uses the bias applied during this step.
        exponent = self.bias_potential[current_bin] / self.D

        self.real_time += (
            np.exp(exponent)
            * self.delta_t
        )

        new_position = last_position + drift + noise

        self.positions.append(
            np.asarray(new_position, dtype=float)
        )

        # Each generated position contributes one force observation.
        self._update_force_estimator(new_position)

        # Reconstruct the profiles after every force observation.
        self.free_energy()
        self.update_bias_potential()

        # Only the newly generated point needs to be checked here.
        self.td.positions = np.asarray(
            [new_position],
            dtype=float,
        )

        return self.td.detect_transition() is not None

    def simulate_steps(self, num_steps=None):
        """
        Perform a batch of integration steps.

        Parameters
        ----------
        num_steps : int or None
            Number of steps in the batch. When None, ``self.num_steps``
            is used.

        Returns
        -------
        int or None
            Global trajectory index of the transition, or None.
        """
        steps = self.num_steps if num_steps is None else num_steps

        if isinstance(steps, bool) or not isinstance(
            steps,
            (int, np.integer),
        ):
            raise TypeError("num_steps must be an integer.")

        if steps <= 0:
            raise ValueError("num_steps must be strictly positive.")

        for _ in range(int(steps)):
            transition_detected = self.simulate_one_step()

            if transition_detected:
                return len(self.positions) - 1

        return None

    def simulate(self, max_iters=1_000_000):
        """
        Run at most ``max_iters`` integration steps.

        Notes
        -----
        ``max_iters`` is retained for compatibility, but it represents
        integration steps, not batches.
        """
        if isinstance(max_iters, bool) or not isinstance(
            max_iters,
            (int, np.integer),
        ):
            raise TypeError("max_iters must be an integer.")

        if max_iters <= 0:
            raise ValueError("max_iters must be strictly positive.")

        transition_index = None

        for _ in range(int(max_iters)):
            transition_detected = self.simulate_one_step()

            if transition_detected:
                transition_index = len(self.positions) - 1
                break

        transition_time = (
            transition_index * self.delta_t
            if transition_index is not None
            else None
        )

        return self.real_time, transition_time

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