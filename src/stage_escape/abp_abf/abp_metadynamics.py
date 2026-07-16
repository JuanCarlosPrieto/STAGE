import numpy as np
from .potential import Potential


class ABPMetaDynamics:
    def __init__(
        self,
        num_steps,
        td,
        delta_t,
        dimension=1,
        D=1.0,
        initial_position=None,
        b=None,
        W=0.1,
        sigma=0.001,
        seed=None,
        rng=None,
    ):
        if seed is not None and rng is not None:
            raise ValueError("Provide either seed or rng, not both.")

        self.num_steps = num_steps + 1
        self.td = td
        self.delta_t = delta_t
        self.dimension = dimension
        self.D = D

        self.initial_position = (
            np.zeros(dimension, dtype=float)
            if initial_position is None
            else np.asarray(initial_position, dtype=float)
        )

        if b is None:
            b = Potential(
                dimension=dimension,
                function=lambda x: 0.0,
                first_derivative=lambda x: np.zeros(
                    dimension,
                    dtype=float,
                ),
                second_derivative=lambda x: np.zeros(
                    (dimension, dimension),
                    dtype=float,
                ),
            )

        self.b = b
        self.centers = []
        self.W = W
        self.sigma = sigma
        self.real_time = 0.0
        self.transition_index = None
        self.weights = [1.0]

        self.seed = seed
        self._external_rng = rng is not None
        self.rng = (
            rng
            if rng is not None
            else np.random.default_rng(seed)
        )

        self.positions = [self.initial_position.copy()]


    def _last_position(self):
        return self.positions[-1]


    def bias_potential_at(self, x):
        from .abp_bias import gaussian_bias_value

        return gaussian_bias_value(
            position=x,
            centers=self.centers,
            height=self.W,
            sigma=self.sigma,
        )


    def bias_potential_prime_at(self, x):
        from .abp_bias import gaussian_bias_gradient

        return gaussian_bias_gradient(
            position=x,
            centers=self.centers,
            height=self.W,
            sigma=self.sigma,
        )


    def simulate_steps(self):
        for _ in range(1, self.num_steps):
            current_position = self._last_position()

            physical_force = self.b.potential_prime_at(
                current_position
            )

            bias_force = self.bias_potential_prime_at(
                current_position
            )

            drift = -(
                physical_force + bias_force
            ) * self.delta_t

            step_size = np.sqrt(
                2.0 * self.D * self.delta_t
            )

            noise = self.rng.standard_normal(
                self.dimension
            )

            step = step_size * noise
            new_position = current_position + drift + step

            bias_value = self.bias_potential_at(new_position)
            weight = np.exp(bias_value / self.D)

            self.real_time += weight * self.delta_t
            self.positions.append(new_position)
            self.weights.append(weight)

            self.td.positions = np.asarray([new_position], dtype=float)

            if self.td.detect_transition() is not None:
                self.transition_index = len(self.positions) - 1
                return self.transition_index

        self.td.positions = np.asarray(
            self.positions[-self.num_steps:],
            dtype=float,
        )

        return None
    

    def simulate(self, max_iters=1_000_000):
        for _ in range(int(max_iters)):
            transition_index = self.simulate_steps()

            self.centers.append(
                self._last_position().copy()
            )

            if transition_index is not None:
                break

        return self.real_time, self.transition_index


    def result(self):
        from .results import ABPResult

        positions = np.asarray(
            self.positions,
            dtype=float,
        )

        centers = np.asarray(
            self.centers,
            dtype=float,
        )

        if centers.size == 0:
            centers = np.empty(
                (0, self.dimension),
                dtype=float,
            )
        else:
            centers = centers.reshape(
                -1,
                self.dimension,
            )

        weights = np.asarray(
            self.weights,
            dtype=float,
        )

        if len(weights) != len(positions):
            raise RuntimeError(
                "ABP produced an inconsistent result: "
                f"{len(positions)} positions and "
                f"{len(weights)} weights."
            )

        return ABPResult(
            method="abp",
            positions=positions,
            delta_t=self.delta_t,
            diffusion=self.D,
            seed=self.seed,
            transition_index=self.transition_index,
            physical_time=self.real_time,
            weights=weights,
            centers=centers,
            bias_height=self.W,
            bias_width=self.sigma,
            metadata={
                "dimension": self.dimension,
                "deposition_batch_size": self.num_steps - 1,
            },
        )


    def run(self, max_iters=1_000_000):
        self.simulate(max_iters=max_iters)
        return self.result()