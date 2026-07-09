import numpy as np
import matplotlib.pyplot as plt

class ABFRealTime:
    def __init__(self, num_steps, td, delta_t, D=1.0, initial_position=None, b=lambda x: 0, bins=100, range=(-3, 3)):
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

    
    def position_to_bin(self, position):
        x = float(np.asarray(position).reshape(-1)[0])
        bin_index = int((x - self.range[0]) / (self.range[1] - self.range[0]) * self.bins)
        return np.clip(bin_index, 0, self.bins - 1)


    def simulate_steps(self):
        rng = np.random.default_rng()  # Use the new random number generator

        for _ in range(1, self.num_steps):
            x = float(np.asarray(self.positions[-1]).reshape(-1)[0])

            if x < self.range[0] or x > self.range[1]:
                print(self.positions[-1])
                raise ValueError("Position out of bounds. Please check the range and initial position.")
            
            drift = (-self.b.potential_prime_at(self.positions[-1]) + self.force_bias[self.position_to_bin(self.positions[-1])]) * self.delta_t  # Drift term based on the current position
            step_size = np.sqrt(2 * self.D * self.delta_t)  # Step size based on diffusion coefficient and time step
            step = step_size * rng.standard_normal(self.dimension)  # Random step from normal distribution
            new_position = self.positions[-1] + step + drift  # Add drift term
            bin_index = self.position_to_bin(self.positions[-1])

            self.real_time += np.exp(
                self.bias_potential[bin_index] / self.D
            ) * self.delta_t
            self.positions.append(new_position)
        
        self.td.positions = np.array(self.positions[-self.num_steps:])
    

    def simulate(self, max_iters=1e6):
        while len(self.positions) < max_iters:
            self.simulate_steps()
            n = self.number_of_copies[self.position_to_bin(self.positions[-1])]  # Number of copies for the current bin
            self.force_bias[self.position_to_bin(self.positions[-1])] *= n / (n + 1)  # Update the force bias based on the number of copies
            self.force_bias[self.position_to_bin(self.positions[-1])] += self.b.potential_prime_at(self.positions[-1]) / (n + 1)
            self.number_of_copies[self.position_to_bin(self.positions[-1])] += 1  # Increment the number of copies for the current bin
            self.free_energy()  # Update the free energy profile after each step

            if self.td.detect_transition() is not None:
                break
        
        escape_index = self.td.detect_transition()
        return self.real_time, escape_index * self.delta_t if escape_index is not None else None

    
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