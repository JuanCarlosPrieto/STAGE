import numpy as np


class ABFRealTime:
    def __init__(
        self,
        deposition_stride,
        td,
        delta_t,
        D=1.0,
        initial_position=None,
        b=lambda x: 0,
        bins=100,
        value_range=(-3, 3),
        seed=None,
        rng=None,
        profile_update_stride=100,
    ):
        if seed is not None and rng is not None:
            raise ValueError("Provide either seed or rng, not both.")

        self.deposition_stride = deposition_stride
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
        self.value_range = value_range  # Range for the biasing potential
        self.free_energy_profile = np.zeros(bins)  # Initialize free energy profile
        self.real_time = 0.0
        
        if (
            isinstance(profile_update_stride, bool)
            or not isinstance(
                profile_update_stride,
                (int, np.integer),
            )
        ):
            raise TypeError(
                "profile_update_stride must be an integer."
            )

        if profile_update_stride <= 0:
            raise ValueError(
                "profile_update_stride must be strictly positive."
            )

        self.profile_update_stride = profile_update_stride
        self.steps_completed = 0
        self.transition_index = None

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
        bin_index = int((x - self.value_range[0]) / (self.value_range[1] - self.value_range[0]) * self.bins)
        return np.clip(bin_index, 0, self.bins - 1)


    def _last_position(self):
        return np.asarray(self.positions[-1])


    def _current_bin(self):
        return self.position_to_bin(self.positions[-1])
    
    
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
        self.steps_completed += 1

        if (
            self.steps_completed
            % self.profile_update_stride
            == 0
        ):
            self.update_profiles()

        return self.td.is_transition(new_position)

    def step(self) -> bool:
        current_position = self._last_position()

        physical_gradient = np.asarray(
            self.b.potential_prime_at(current_position),
            dtype=float,
        )
        bias_gradient = self.bias_potential_prime_at(
            current_position
        )

        drift = -(
            physical_gradient + bias_gradient
        ) * self.delta_t

        noise = (
            np.sqrt(2.0 * self.D * self.delta_t)
            * self.rng.standard_normal(self.dimension)
        )

        new_position = (
            current_position
            + drift
            + noise
        )

        bias_value = self.bias_potential_at(
            new_position
        )
        weight = np.exp(bias_value / self.D)

        if not np.isfinite(weight):
            raise FloatingPointError(
                "The ABP reweighting factor is not finite."
            )

        self.positions.append(new_position)
        self.weights.append(weight)
        self.real_time += weight * self.delta_t
        self.steps_completed += 1

        if self.td.is_transition(new_position):
            self.transition_index = (
                len(self.positions) - 1
            )
            return True

        if (
            self.steps_completed
            % self.deposition_stride
            == 0
        ):
            self.centers.append(
                new_position.copy()
            )

        return False

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

        for _ in range(int(max_iters)):
            transition_detected = self.simulate_one_step()

            if transition_detected:
                self.transition_index = len(self.positions) - 1
                break

        transition_time = (
            self.transition_index * self.delta_t
            if self.transition_index is not None
            else None
        )

        self.update_profiles()

        return self.real_time, transition_time
    
    def update_profiles(self):
        from .abf_profiles import reconstruct_abf_profiles

        (
            _,
            self.free_energy_profile,
            self.bias_potential,
        ) = reconstruct_abf_profiles(
            force_bias=self.force_bias,
            value_range=self.range,
        )

        return (
            self.free_energy_profile,
            self.bias_potential,
        )
    

    def result(self):
        from .abf_profiles import reconstruct_abf_profiles
        from .results import ABFResult

        (
            bin_edges,
            free_energy,
            bias_potential,
        ) = reconstruct_abf_profiles(
            force_bias=self.force_bias,
            value_range=self.range,
        )

        positions = np.asarray(
            self.positions,
            dtype=float,
        )

        return ABFResult(
            method="abf",
            positions=positions,
            delta_t=self.delta_t,
            diffusion=self.D,
            seed=self.seed,
            transition_index=self.transition_index,
            physical_time=self.real_time,
            force_bias=np.asarray(
                self.force_bias,
                dtype=float,
            ).copy(),
            visit_counts=np.asarray(
                self.number_of_copies,
                dtype=float,
            ).copy(),
            bin_edges=bin_edges,
            free_energy=free_energy,
            bias_potential=bias_potential,
            metadata={
                "range": tuple(self.range),
                "bins": self.bins,
                "profile_update_stride": (
                    self.profile_update_stride
                ),
            },
        )


    def run(self, max_steps=1_000_000):
        for _ in range(max_steps):
            if self.step():
                break

        self.update_profiles()
        return self.result()
    