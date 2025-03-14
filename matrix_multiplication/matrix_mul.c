#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define MATRIX_SIZE 1000
#define MAX_THREADS 8

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
    
    // Run sequential version first
    double start_time = omp_get_wtime();
    sequential_multiply(A, B, C, size);
    double seq_time = omp_get_wtime() - start_time;
    printf("Sequential time: %f seconds\n", seq_time);
    
    // Test with different numbers of threads
    printf("\nThread count, Time(s), Speedup\n");
    for (int threads = 1; threads <= MAX_THREADS; threads *= 2) {
        omp_set_num_threads(threads);
        
        start_time = omp_get_wtime();
        parallel_multiply(A, B, C, size);
        double par_time = omp_get_wtime() - start_time;
        
        double speedup = seq_time / par_time;
        printf("%d, %f, %f\n", threads, par_time, speedup);
    }
    
    // Free memory
    free(A);
    free(B);
    free(C);
    
    return 0;
} 