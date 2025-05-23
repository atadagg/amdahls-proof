import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

def ensure_experiment_dir(experiment_num):
    """Create and return the experiment directory path"""
    exp_dir = f"results/experiment_{experiment_num}"
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def amdahls_law(n, p):
    """
    Calculate theoretical speedup according to Amdahl's Law
    n: number of processors
    p: fraction of program that is parallelizable (0 to 1)
    """
    return 1 / ((1 - p) + p/n)

def plot_amdahls_law(experiment_num):
    # Number of processors (x-axis)
    n = np.linspace(1, 16, 100)
    
    # Different parallelizable fractions
    fractions = [0.5, 0.75, 0.9, 0.95, 0.99]
    colors = ['r', 'g', 'b', 'c', 'm']
    
    plt.figure(figsize=(10, 6))
    
    # Plot theoretical curves
    for p, color in zip(fractions, colors):
        speedup = [amdahls_law(i, p) for i in n]
        plt.plot(n, speedup, color=color, label=f'p={p:.2f}')
    
    # Formatting
    plt.title("Amdahl's Law: Theoretical Speedup vs. Number of Processors")
    plt.xlabel('Number of Processors (n)')
    plt.ylabel('Speedup')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot in experiment directory
    exp_dir = ensure_experiment_dir(experiment_num)
    plt.savefig(os.path.join(exp_dir, 'amdahls_law_theoretical.png'))
    plt.close()

def plot_experimental_results(experimental_speedups, algorithm_name, p_value, output_dir, filename_base):
    """
    Plot experimental results against Amdahl's Law
    experimental_speedups: list of (n_threads, speedup) tuples
    algorithm_name: name of the algorithm for the plot title
    p_value: theoretical parallelizable fraction
    output_dir: directory to save the plot
    filename_base: base name for the output file
    """
    if not experimental_speedups:
        print(f"Warning: No data for {algorithm_name}")
        return
        
    n_threads, speedups = zip(*experimental_speedups)
    n_threads = np.array(n_threads)
    speedups = np.array(speedups)
    
    plt.figure(figsize=(12, 8))
    
    # Plot theoretical curve with given p_value
    n = np.linspace(1, max(n_threads), 100)
    theoretical = [amdahls_law(i, p_value) for i in n]
    plt.plot(n, theoretical, 'b-', label=f'Theoretical (p={p_value:.2f})')
    
    # Plot experimental data
    plt.plot(n_threads, speedups, 'ro-', label='Experimental')
    
    # Add ideal speedup line
    plt.plot(n, n, 'g--', label='Ideal Speedup', alpha=0.5)
    
    # Formatting
    plt.title(f"Amdahl's Law: {algorithm_name} Implementation", fontsize=14)
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Set y-axis to start from 1
    plt.ylim(bottom=1)
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'{filename_base}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {plot_path}")

def plot_amdahls_law_theoretical_overview(output_dir):
    """Plot theoretical Amdahl's Law curves for different p values"""
    n = np.linspace(1, 16, 100)
    
    # Different parallelizable fractions
    fractions = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    
    plt.figure(figsize=(12, 8))
    
    # Plot theoretical curves
    for p, color in zip(fractions, colors):
        speedup = [amdahls_law(i, p) for i in n]
        plt.plot(n, speedup, color=color, label=f'p={p:.2f}')
    
    # Add ideal speedup line
    plt.plot(n, n, 'k--', label='Ideal Speedup', alpha=0.5)
    
    # Formatting
    plt.title("Amdahl's Law: Theoretical Speedup vs. Number of Processors", fontsize=14)
    plt.xlabel('Number of Processors (n)', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Set y-axis to start from 1
    plt.ylim(bottom=1)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'amdahls_law_theoretical.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Theoretical overview plot saved to: {plot_path}")

if __name__ == "__main__":
    # Example usage with experimental data:
    plot_amdahls_law(1)
    
    matrix_speedups = [
        (1, 1.0),    # baseline
        (2, 1.8),    # example speedup with 2 threads
        (4, 3.2),    # example speedup with 4 threads
        (8, 5.1),    # example speedup with 8 threads
    ]
    
    plot_experimental_results(matrix_speedups, "Matrix Multiplication", 0.9, ensure_experiment_dir(1), "amdahls_law_matrix_multiplication") 