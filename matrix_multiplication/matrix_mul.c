#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define MATRIX_SIZE 1000
#define NUM_THREADS 4

// Initialize matrix with random values
void init_matrix(double* matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i * size + j] = rand() % 100;
        }
    }
}

// Sequential matrix multiplication
void sequential_multiply(double* A, double* B, double* C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double sum = 0.0;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

// Parallel matrix multiplication using OpenMP
void parallel_multiply(double* A, double* B, double* C, int size) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double sum = 0.0;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

int main() {
    int size = MATRIX_SIZE;
    double *A, *B, *C;
    
    // Allocate memory
    A = (double*)malloc(size * size * sizeof(double));
    B = (double*)malloc(size * size * sizeof(double));
    C = (double*)malloc(size * size * sizeof(double));
    
    if (!A || !B || !C) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    // Initialize matrices
    srand(time(NULL));
    init_matrix(A, size);
    init_matrix(B, size);
    
    // Set number of threads
    omp_set_num_threads(NUM_THREADS);
    
    // Sequential multiplication
    double start_time = omp_get_wtime();
    sequential_multiply(A, B, C, size);
    double seq_time = omp_get_wtime() - start_time;
    
    // Parallel multiplication
    start_time = omp_get_wtime();
    parallel_multiply(A, B, C, size);
    double par_time = omp_get_wtime() - start_time;
    
    // Print results
    printf("Matrix size: %d x %d\n", size, size);
    printf("Number of threads: %d\n", NUM_THREADS);
    printf("Sequential time: %f seconds\n", seq_time);
    printf("Parallel time: %f seconds\n", par_time);
    printf("Speedup: %f\n", seq_time / par_time);
    
    // Free memory
    free(A);
    free(B);
    free(C);
    
    return 0;
} 