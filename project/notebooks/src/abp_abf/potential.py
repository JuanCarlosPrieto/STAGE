import numpy as np

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