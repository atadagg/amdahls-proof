import subprocess
import csv
import os
from visualize_amdahl import plot_experimental_results, plot_amdahls_law_theoretical_overview

NUM_RUNS = 5 
OUTPUT_DIR_BASE = "results"

P_VALUES = {
    "matrix_mul": 0.93,  # Example from your graph
    "dijkstra": 0.56,    # Example from your graph
    "tsp_solver": 0.00,  # Example from your graph
    "heterogeneous_workload_exe": 0.75  # Estimated p-value for heterogeneous workload
}

EXECUTABLES = {
    "Matrix Multiplication": "matrix_mul",
    "Dijkstra": "dijkstra",
    "TSP": "tsp_solver",
    "Heterogeneous Workload": "heterogeneous_workload_exe"
}
# Define thread counts to be tested by C programs (if not dynamic in C)
# Your C programs seem to output multiple thread counts, so this might not be needed here
# THREAD_COUNTS = [1, 2, 4, 8]

def run_single_executable_pass(executable_path):
    """Runs a single C executable once and parses its output for multiple thread results."""
    # Ensure the C executable itself runs tests for 1, 2, 4, 8 threads
    # and outputs lines like: "threads,time,speedup"
    try:
        result = subprocess.run([f'./{executable_path}'], capture_output=True, text=True, check=True, timeout=300) # Added check=True and timeout
    except subprocess.CalledProcessError as e:
        print(f"Error running {executable_path}: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        print(f"Timeout running {executable_path}")
        return None

    lines = result.stdout.strip().split('\n')
    parsed_data = [] # Will store list of (threads, time_val, speedup_val)
    header_found = False
    for line in lines:
        if 'Thread count' in line or 'threads,time,speedup' in line.lower(): # Make header check more robust
            header_found = True
            continue
        if header_found and ',' in line:
            try:
                threads, time_val, speedup_val = map(float, line.strip().split(','))
                # Speedup should ideally be calculated based on the 1-thread time from *this current set of runs*
                # If C program calculates it based on an internal 1-thread run, ensure that's consistent.
                parsed_data.append((int(threads), time_val, speedup_val))
            except ValueError:
                print(f"Warning: Could not parse line: {line.strip()} in output of {executable_path}")
                continue
    if not header_found:
        print(f"Warning: Header not found in output of {executable_path}")
    if not parsed_data:
        print(f"Warning: No data parsed from {executable_path}. Output:\n{result.stdout}")
        return None
    return parsed_data


def run_experiment_multiple_times(executable_name, num_runs=NUM_RUNS):
    """
    Runs the C executable multiple times, collects data for all thread counts,
    and averages the times.
    Assumes the C executable tests multiple thread counts (1, 2, 4, 8) in a single run
    and outputs "threads,time,speedup".
    """
    all_runs_data = {}  # Key: thread_count, Value: list of (time_val, speedup_val)

    print(f"  Running {executable_name} {num_runs} times...")
    for i in range(num_runs):
        print(f"    Run {i+1}/{num_runs}...")
        single_pass_results = run_single_executable_pass(executable_name)
        if single_pass_results is None:
            print(f"    Run {i+1} failed for {executable_name}. Skipping this run.")
            continue

        for threads, time_val, speedup_val_from_c in single_pass_results:
            if threads not in all_runs_data:
                all_runs_data[threads] = {'times': [], 'c_speedups': []}
            all_runs_data[threads]['times'].append(time_val)
            all_runs_data[threads]['c_speedups'].append(speedup_val_from_c) # Store C-calculated speedup if needed for comparison

    if not all_runs_data:
        print(f"  No data collected for {executable_name} after {num_runs} runs.")
        return []

    # Calculate averages
    averaged_results = [] # List of (threads, avg_time, calculated_speedup)
    
    # Get the 1-thread average time for calculating our own speedups
    baseline_time_1_thread = 0
    if 1 in all_runs_data and all_runs_data[1]['times']:
        baseline_time_1_thread = sum(all_runs_data[1]['times']) / len(all_runs_data[1]['times'])
    else:
        print(f"  Warning: No 1-thread data found for {executable_name} to calculate speedup. Speedup will be 0 or based on C output.")
        # Fallback or error handling needed if 1-thread data is essential and missing

    for threads, data in sorted(all_runs_data.items()):
        if not data['times']:
            continue
        avg_time = sum(data['times']) / len(data['times'])
        
        # Calculate speedup based on our averaged 1-thread baseline
        # This is generally more robust than relying on C's internal 1-thread run for each speedup calculation
        calculated_speedup = baseline_time_1_thread / avg_time if baseline_time_1_thread > 0 and avg_time > 0 else 0
        
        # You can also average the speedups reported by C, but recalculating is often better
        # avg_c_speedup = sum(data['c_speedups']) / len(data['c_speedups']) if data['c_speedups'] else 0
        
        averaged_results.append((threads, avg_time, calculated_speedup))
        print(f"    {threads} threads: Avg Time={avg_time:.4f}s, Speedup={calculated_speedup:.2f}x")

    return averaged_results


def get_next_experiment_number(base_dir):
    """Get the next experiment number by checking existing directories"""
    exp_num = 1
    while os.path.exists(os.path.join(base_dir, f"experiment_{exp_num}")):
        exp_num += 1
    return exp_num

def save_results_to_csv(data, filename):
    """Saves [ (threads, avg_time, speedup), ... ] to a CSV file."""
    if not data:
        print(f"No data to save to {filename}")
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Threads', 'Average Time (s)', 'Speedup'])
        for row in data:
            writer.writerow(row)
    print(f"  Results saved to {filename}")


def main():
    # Create base results directory if it doesn't exist
    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

    experiment_num = get_next_experiment_number(OUTPUT_DIR_BASE)
    current_experiment_dir = os.path.join(OUTPUT_DIR_BASE, f"experiment_{experiment_num}")
    os.makedirs(current_experiment_dir, exist_ok=True)
    print(f"\nStarting experiment {experiment_num}. Results will be in: {current_experiment_dir}")

    # Compile all programs
    print("Compiling C programs...")
    try:
        subprocess.run(['make', 'clean'], check=True)
        subprocess.run(['make'], check=True)
        print("Compilation successful.")
    except subprocess.CalledProcessError as e:
        print(f"Make command failed: {e}")
        return # Exit if compilation fails

    # Generate theoretical Amdahl's Law overview plot (like Graph 1)
    # This plot shows Amdahl's for various p values, not tied to specific experiments yet.
    # Ensure 'plot_amdahls_law_theoretical_overview' from visualize_amdahl.py does this.
    print("Generating theoretical Amdahl's Law overview plot...")
    plot_amdahls_law_theoretical_overview(current_experiment_dir) # Pass directory to save the plot

    for algorithm_name, executable_file in EXECUTABLES.items():
        print(f"\nRunning experiment for: {algorithm_name} (executable: {executable_file})")
        
        # Run experiment multiple times and get averaged data (threads, avg_time, calculated_speedup)
        # The `calculated_speedup` here is based on the average 1-thread time from these runs.
        experiment_data = run_experiment_multiple_times(executable_file, NUM_RUNS)

        if not experiment_data:
            print(f"Skipping plotting and CSV saving for {algorithm_name} due to no data.")
            continue
            
        # Extract just (threads, speedup) for the plotting function if it expects that
        speedups_for_plot = [(d[0], d[2]) for d in experiment_data] 

        # Plot experimental results against theoretical for this specific algorithm
        p_value = P_VALUES[executable_file]
        plot_filename_base = f"{executable_file.replace('_', '-')}_speedup"
        plot_experimental_results(
            experimental_speedups=speedups_for_plot,
            algorithm_name=algorithm_name,
            p_value=p_value, # Pass the p_value for this algorithm
            output_dir=current_experiment_dir, # Pass directory to save the plot
            filename_base=plot_filename_base
        )
        
        # Save the detailed (threads, avg_time, speedup) data to CSV
        csv_filename = os.path.join(current_experiment_dir, f"{executable_file.replace('_', '-')}_data.csv")
        save_results_to_csv(experiment_data, csv_filename)

    print(f"\nExperiments completed. Results saved in {current_experiment_dir}/")

if __name__ == "__main__":
    main()