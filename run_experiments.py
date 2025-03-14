import subprocess
import csv
import os
import time
from visualize_amdahl import plot_experimental_results, plot_amdahls_law

def run_experiment(executable):
    """Run the experiment and parse its output"""
    result = subprocess.run([f'./{executable}'], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    
    # Skip header lines and parse the CSV data
    speedups = []
    header_found = False
    
    for line in lines:
        if 'Thread count' in line:
            header_found = True
            continue
        if header_found and ',' in line:
            try:
                threads, time, speedup = map(float, line.strip().split(','))
                speedups.append((int(threads), speedup))
            except ValueError:
                continue
    
    return speedups

def get_next_experiment_number():
    """Get the next experiment number by checking existing directories"""
    exp_num = 1
    while os.path.exists(f"results/experiment_{exp_num}"):
        exp_num += 1
    return exp_num

def main():
    # Get the next experiment number
    experiment_num = get_next_experiment_number()
    print(f"\nStarting experiment {experiment_num}")
    
    # First, compile all programs
    subprocess.run(['make', 'clean'])
    subprocess.run(['make'])
    
    # Generate theoretical Amdahl's Law plot
    plot_amdahls_law(experiment_num)
    
    # Run matrix multiplication experiment
    print("Running matrix multiplication experiment...")
    matrix_speedups = run_experiment('matrix_mul')
    plot_experimental_results(matrix_speedups, "Matrix Multiplication", experiment_num)
    
    # Run Dijkstra's algorithm experiment
    print("Running Dijkstra's algorithm experiment...")
    dijkstra_speedups = run_experiment('dijkstra')
    plot_experimental_results(dijkstra_speedups, "Dijkstra", experiment_num)
    
    # Run TSP experiment
    print("Running TSP experiment...")
    tsp_speedups = run_experiment('tsp_solver')
    plot_experimental_results(tsp_speedups, "TSP", experiment_num)
    
    print(f"\nExperiments completed. Results saved in results/experiment_{experiment_num}/")

if __name__ == "__main__":
    main() 