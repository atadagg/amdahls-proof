import numpy as np
import matplotlib.pyplot as plt
import os

def ensure_experiment_dir(experiment_num):
    """Create and return the experiment directory path"""
    exp_dir = f"results/experiment_{experiment_num}"
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def amdahls_law(p, n):
    """
    Calculate theoretical speedup according to Amdahl's Law
    p: fraction of program that is parallelizable (0 to 1)
    n: number of processors
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
        speedup = [amdahls_law(p, i) for i in n]
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

def plot_experimental_results(measured_speedups, algorithm_name, experiment_num):
    """
    Plot experimental results against Amdahl's Law
    measured_speedups: list of (n_threads, speedup) tuples
    """
    if not measured_speedups:
        print(f"Warning: No data for {algorithm_name}")
        return
        
    n_threads, speedups = zip(*measured_speedups)
    
    # Fit Amdahl's law to the experimental data
    max_speedup = max(speedups)
    max_threads = n_threads[speedups.index(max_speedup)]
    p_estimate = (max_speedup - 1) / (max_speedup - 1/max_threads)
    p_estimate = min(p_estimate, 0.99)  # Cap at 0.99 for realism
    
    plt.figure(figsize=(10, 6))
    
    # Plot theoretical curve with estimated p
    n = np.linspace(1, max(n_threads), 100)
    theoretical = [amdahls_law(p_estimate, i) for i in n]
    plt.plot(n, theoretical, 'b-', label=f'Theoretical (p≈{p_estimate:.2f})')
    
    # Plot experimental data
    plt.plot(n_threads, speedups, 'ro-', label='Experimental')
    
    # Formatting
    plt.title(f"Amdahl's Law: {algorithm_name} Implementation")
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot in experiment directory
    exp_dir = ensure_experiment_dir(experiment_num)
    plt.savefig(os.path.join(exp_dir, f'amdahls_law_{algorithm_name.lower().replace(" ", "_")}.png'))
    plt.close()
    
    # Save the raw data
    with open(os.path.join(exp_dir, f'{algorithm_name.lower().replace(" ", "_")}_data.csv'), 'w') as f:
        f.write("Threads,Speedup\n")
        for thread, speedup in measured_speedups:
            f.write(f"{thread},{speedup}\n")

if __name__ == "__main__":
    # Example usage with experimental data:
    plot_amdahls_law(1)
    
    matrix_speedups = [
        (1, 1.0),    # baseline
        (2, 1.8),    # example speedup with 2 threads
        (4, 3.2),    # example speedup with 4 threads
        (8, 5.1),    # example speedup with 8 threads
    ]
    
    plot_experimental_results(matrix_speedups, "Matrix Multiplication", 1) 