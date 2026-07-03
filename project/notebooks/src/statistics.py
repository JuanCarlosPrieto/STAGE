import matplotlib.pyplot as plt
import numpy as np

class Statistics:
    @staticmethod
    def plot_mean_times(x, mean_times, std_devs, xlabel='X-axis', ylabel='Mean Times', title='Mean Times with Standard Deviation'):
        mean_times = np.array(mean_times)
        std_devs = np.array(std_devs)

        plt.figure(figsize=(10, 6))
        plt.errorbar(x, mean_times, yerr=std_devs, fmt='-o', ecolor='r', capsize=5, label='Mean Times')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()


    @staticmethod
    def plot_histogram(data, bins=30, xlabel='Data', ylabel='Frequency', title='Histogram'):
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()
        plt.show()


    @staticmethod
    def plot_histogram_with_fit(data, bins=30, xlabel='Data', ylabel='Frequency', title='Histogram with Exponential Fit', t_min=0):
        plt.figure(figsize=(10, 6))
        counts, bin_edges, _ = plt.hist(data, bins=bins, alpha=0.7, color='blue', edgecolor='black', density=True)

        # Fit an exponential distribution to the data
        lambda_param = Statistics.adapt_exp_distribution(data, t_min=t_min)
        x = np.linspace(min(data), max(data), 100)
        pdf = lambda_param * np.exp(-lambda_param * (x - t_min))

        plt.plot(x, pdf, 'r-', lw=2, label=f'Exponential Fit (λ={lambda_param:.2f})')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()

    
    @staticmethod
    def adapt_exp_distribution(data, t_min=0):
        """
        Adapt the exponential distribution to the given data.
        
        Parameters:
        - data: The input data to fit the exponential distribution.
        - t_min: The minimum time threshold for the distribution.
        
        Returns:
        - lambda_param: The estimated rate parameter of the exponential distribution.
        """
        # Filter data based on t_min
        filtered_data = np.array([x for x in data if x >= t_min])
        
        if len(filtered_data) == 0:
            raise ValueError("No data points are greater than or equal to t_min.")
        
        # Estimate the rate parameter (lambda) of the exponential distribution
        lambda_param = 1 / (np.mean(filtered_data) - t_min)
        
        return lambda_param