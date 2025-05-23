#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <limits.h>
#include <unistd.h>

// Constants
#define MATRIX_SIZE 1000
#define GRAPH_SIZE 1000
#define NUM_CITIES 15
#define MAX_THREADS 8

// Matrix multiplication structures
typedef struct {
    double* A;
    double* B;
    double* C;
    int size;
} MatrixWork;

// Dijkstra structures
typedef struct {
    int* graph;
    int* dist;
    int size;
    int source;
} GraphWork;

// TSP structures
typedef struct {
    int x;
    int y;
} City;

typedef struct {
    City* cities;
    int num_cities;
} TSPWork;

// Matrix multiplication functions
void init_matrix(double* matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = (double)rand() / RAND_MAX;
    }
}

void parallel_matrix_multiply(double* A, double* B, double* C, int size) {
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

// Dijkstra functions
void init_graph(int* graph, int size) {
    for (int i = 0; i < size * size; i++) {
        if (i % (size + 1) == 0) {
            graph[i] = 0;  // Diagonal elements
        } else {
            graph[i] = rand() % 100 + 1;  // Random weights
        }
    }
}

int min_distance(int* dist, int* sptSet, int size) {
    int min = INT_MAX;
    int min_index = -1;
    
    #pragma omp parallel 
    {
        int local_min = INT_MAX;
        int local_min_index = -1;
        
        #pragma omp for nowait
        for (int v = 0; v < size; v++) {
            if (sptSet[v] == 0 && dist[v] <= local_min) {
                local_min = dist[v];
                local_min_index = v;
            }
        }
        
        #pragma omp critical
        {
            if (local_min < min || (local_min == min && local_min_index < min_index)) {
                min = local_min;
                min_index = local_min_index;
            }
        }
    }
    
    return min_index;
}

void parallel_dijkstra(int* graph, int* dist, int size, int src) {
    int* sptSet = (int*)calloc(size, sizeof(int));
    
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        dist[i] = INT_MAX;
        sptSet[i] = 0;
    }
    
    dist[src] = 0;
    
    for (int count = 0; count < size - 1; count++) {
        int u = min_distance(dist, sptSet, size);
        sptSet[u] = 1;
        
        #pragma omp parallel for
        for (int v = 0; v < size; v++) {
            if (!sptSet[v] && graph[u * size + v] && dist[u] != INT_MAX &&
                dist[u] + graph[u * size + v] < dist[v]) {
                dist[v] = dist[u] + graph[u * size + v];
            }
        }
    }
    
    free(sptSet);
}

// TSP functions
void init_cities(City* cities, int num_cities) {
    for (int i = 0; i < num_cities; i++) {
        cities[i].x = rand() % 100;
        cities[i].y = rand() % 100;
    }
}

int calculate_distance(City a, City b) {
    return abs(a.x - b.x) + abs(a.y - b.y);
}

int parallel_tsp(City* cities, int num_cities) {
    int* visited = (int*)calloc(num_cities, sizeof(int));
    int current = 0;
    int total_distance = 0;
    visited[0] = 1;
    
    for (int i = 1; i < num_cities; i++) {
        int global_min_dist = INT_MAX;
        int next_city = -1;
        
        #pragma omp parallel 
        {
            int local_min_dist = INT_MAX;
            
            #pragma omp for nowait
            for (int j = 0; j < num_cities; j++) {
                if (!visited[j]) {
                    int dist = calculate_distance(cities[current], cities[j]);
                    if (dist < local_min_dist) {
                        local_min_dist = dist;
                    }
                }
            }
            
            #pragma omp critical
            {
                if (local_min_dist < global_min_dist) {
                    global_min_dist = local_min_dist;
                }
            }
        }
        
        for (int j = 0; j < num_cities; j++) {
            if (!visited[j] && calculate_distance(cities[current], cities[j]) == global_min_dist) {
                next_city = j;
                break;
            }
        }
        
        if (next_city == -1) break;
        
        total_distance += global_min_dist;
        current = next_city;
        visited[current] = 1;
    }
    
    total_distance += calculate_distance(cities[current], cities[0]);
    free(visited);
    return total_distance;
}

// Workload distribution function
void process_workload(int num_threads) {
    // Initialize work structures
    MatrixWork matrix_work = {
        .A = (double*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double)),
        .B = (double*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double)),
        .C = (double*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double)),
        .size = MATRIX_SIZE
    };
    
    GraphWork graph_work = {
        .graph = (int*)malloc(GRAPH_SIZE * GRAPH_SIZE * sizeof(int)),
        .dist = (int*)malloc(GRAPH_SIZE * sizeof(int)),
        .size = GRAPH_SIZE,
        .source = 0
    };
    
    TSPWork tsp_work = {
        .cities = (City*)malloc(NUM_CITIES * sizeof(City)),
        .num_cities = NUM_CITIES
    };
    
    // Initialize data
    init_matrix(matrix_work.A, MATRIX_SIZE);
    init_matrix(matrix_work.B, MATRIX_SIZE);
    init_graph(graph_work.graph, GRAPH_SIZE);
    init_cities(tsp_work.cities, NUM_CITIES);
    
    // Set number of threads
    omp_set_num_threads(num_threads);
    
    // Process all workloads in parallel
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            printf("Thread %d: Starting Matrix Multiplication\n", omp_get_thread_num());
            parallel_matrix_multiply(matrix_work.A, matrix_work.B, matrix_work.C, MATRIX_SIZE);
            printf("Thread %d: Finished Matrix Multiplication\n", omp_get_thread_num());
        }
        
        #pragma omp section
        {
            printf("Thread %d: Starting Dijkstra's Algorithm\n", omp_get_thread_num());
            parallel_dijkstra(graph_work.graph, graph_work.dist, GRAPH_SIZE, graph_work.source);
            printf("Thread %d: Finished Dijkstra's Algorithm\n", omp_get_thread_num());
        }
        
        #pragma omp section
        {
            printf("Thread %d: Starting TSP\n", omp_get_thread_num());
            int tsp_distance = parallel_tsp(tsp_work.cities, NUM_CITIES);
            printf("Thread %d: Finished TSP with distance %d\n", omp_get_thread_num(), tsp_distance);
        }
    }
    
    // Cleanup
    free(matrix_work.A);
    free(matrix_work.B);
    free(matrix_work.C);
    free(graph_work.graph);
    free(graph_work.dist);
    free(tsp_work.cities);
}

int main() {
    srand(time(NULL));
    
    // Run sequential version first
    double start_time = omp_get_wtime();
    process_workload(1);  // Sequential run
    double seq_time = omp_get_wtime() - start_time;
    printf("\nSequential time: %f seconds\n", seq_time);
    
    // Test with different numbers of threads
    printf("\nThread count, Time(s), Speedup\n");
    for (int threads = 1; threads <= MAX_THREADS; threads *= 2) {
        start_time = omp_get_wtime();
        process_workload(threads);
        double par_time = omp_get_wtime() - start_time;
        
        double speedup = seq_time / par_time;
        printf("%d, %f, %f\n", threads, par_time, speedup);
    }
    
    return 0;
} 