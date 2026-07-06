import numpy as np
import matplotlib.pyplot as plt

class ABF:
    def __init__(self, num_steps, delta_t, D=1.0, initial_position=None, b=lambda x: 0, bins=100, range=(-3, 3)):
        self.num_steps = num_steps
        self.delta_t = delta_t
        self.dimension = 1  # Dimension of the Brownian motion
        self.D = D  # Diffusion coefficient
        self.initial_position = initial_position if initial_position is not None else np.zeros(self.dimension)
        self.b = b  # Drift function
        self.force_bias = np.zeros(bins)  # Initialize force bias for each dimension and bin
        self.number_of_copies = np.zeros(bins)  # Initialize number of copies for each dimension and bin
        self.bins = bins  # Number of bins for the biasing potential
        self.range = range  # Range for the biasing potential
        self.free_energy_profile = np.zeros(bins)  # Initialize free energy profile

        if initial_position is None:
            initial_position = np.zeros(self.dimension)  # Default initial position is the origin

        self.positions = [initial_position]  # Start at the specified initial position

    
    def position_to_bin(self, position):
        # Map the position to a bin index based on the specified range and number of bins
        bin_index = int((position[0] - self.range[0]) / (self.range[1] - self.range[0]) * self.bins)
        return np.clip(bin_index, 0, self.bins - 1)  # Ensure the bin index is within valid bounds


    def simulate(self):
        rng = np.random.default_rng()  # Use the new random number generator

        for _ in range(1, self.num_steps):
            if self.positions[-1] < self.range[0] or self.positions[-1] > self.range[1]:
                raise ValueError("Position out of bounds. Please check the range and initial position.")
            
            drift = (-self.b.potential_prime_at(self.positions[-1]) + self.force_bias[self.position_to_bin(self.positions[-1])]) * self.delta_t  # Drift term based on the current position
            n = self.number_of_copies[self.position_to_bin(self.positions[-1])]  # Number of copies for the current bin
            self.force_bias[self.position_to_bin(self.positions[-1])] *= n / (n + 1)  # Update the force bias based on the number of copies
            self.force_bias[self.position_to_bin(self.positions[-1])] += self.b.potential_prime_at(self.positions[-1]) / (n + 1)
            self.number_of_copies[self.position_to_bin(self.positions[-1])] += 1  # Increment the number of copies for the current bin

            step_size = np.sqrt(2 * self.D * self.delta_t)  # Step size based on diffusion coefficient and time step
            step = step_size * rng.standard_normal(self.dimension)  # Random step from normal distribution
            new_position = self.positions[-1] + step + drift  # Add drift term
            self.positions.append(new_position)

    
    def free_energy(self):
        # Calculate the free energy profile based on the force bias and number of copies
        free_energy = np.zeros(self.bins)
        d_x = (self.range[1] - self.range[0]) / self.bins  # Width of each bin
        free_energy[0] = self.force_bias[0] * d_x
        for bin_index in range(1, self.bins):
            free_energy[bin_index] = free_energy[bin_index - 1] + self.force_bias[bin_index] * d_x
        
        self.free_energy_profile = free_energy
        return self.free_energy_profile


    def plot_free_energy(self):
        bin_centers = np.linspace(self.range[0], self.range[1], self.bins)  # Calculate bin centers for plotting
        plt.plot(bin_centers, self.free_energy_profile)  # Plot the free energy profile
        plt.xlabel('Position')  # Label for x-axis
        plt.ylabel('Free Energy')  # Label for y-axis
        plt.title('Free Energy Profile')  # Title of the plot
        plt.grid()  # Add grid to the plot
        plt.show()  # Display the plot