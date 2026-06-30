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