#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
#include <time.h>

#define V 1000  // Number of vertices
#define INF INT_MAX
#define NUM_THREADS 4

// Function to find the vertex with minimum distance value
int minDistance(int dist[], int sptSet[]) {
    int min = INF, min_index;
    
    #pragma omp parallel for reduction(min:min_index)
    for (int v = 0; v < V; v++) {
        if (sptSet[v] == 0 && dist[v] <= min) {
            min = dist[v];
            min_index = v;
        }
    }
    return min_index;
}

// Sequential Dijkstra implementation
void sequentialDijkstra(int graph[V][V], int src, int dist[V]) {
    int sptSet[V];
    
    // Initialize all distances as INFINITE and stpSet[] as false
    for (int i = 0; i < V; i++) {
        dist[i] = INF;
        sptSet[i] = 0;
    }
    
    // Distance of source vertex from itself is always 0
    dist[src] = 0;
    
    // Find shortest path for all vertices
    for (int count = 0; count < V - 1; count++) {
        int u = minDistance(dist, sptSet);
        
        // Mark the picked vertex as processed
        sptSet[u] = 1;
        
        // Update dist value of the adjacent vertices
        for (int v = 0; v < V; v++) {
            if (!sptSet[v] && graph[u][v] && dist[u] != INF &&
                dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }
}

// Parallel Dijkstra implementation
void parallelDijkstra(int graph[V][V], int src, int dist[V]) {
    int sptSet[V];
    
    // Initialize all distances as INFINITE and stpSet[] as false
    #pragma omp parallel for
    for (int i = 0; i < V; i++) {
        dist[i] = INF;
        sptSet[i] = 0;
    }
    
    // Distance of source vertex from itself is always 0
    dist[src] = 0;
    
    // Find shortest path for all vertices
    for (int count = 0; count < V - 1; count++) {
        int u = minDistance(dist, sptSet);
        
        // Mark the picked vertex as processed
        sptSet[u] = 1;
        
        // Update dist value of the adjacent vertices
        #pragma omp parallel for
        for (int v = 0; v < V; v++) {
            if (!sptSet[v] && graph[u][v] && dist[u] != INF &&
                dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }
}

// Generate random graph
void generateGraph(int graph[V][V]) {
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (i == j) {
                graph[i][j] = 0;
            } else {
                graph[i][j] = rand() % 100 + 1;  // Random weights between 1 and 100
            }
        }
    }
}

int main() {
    int graph[V][V];
    int dist[V];
    
    // Generate random graph
    srand(time(NULL));
    generateGraph(graph);
    
    // Set number of threads
    omp_set_num_threads(NUM_THREADS);
    
    // Sequential Dijkstra
    double start_time = omp_get_wtime();
    sequentialDijkstra(graph, 0, dist);
    double seq_time = omp_get_wtime() - start_time;
    
    // Parallel Dijkstra
    start_time = omp_get_wtime();
    parallelDijkstra(graph, 0, dist);
    double par_time = omp_get_wtime() - start_time;
    
    // Print results
    printf("Number of vertices: %d\n", V);
    printf("Number of threads: %d\n", NUM_THREADS);
    printf("Sequential time: %f seconds\n", seq_time);
    printf("Parallel time: %f seconds\n", par_time);
    printf("Speedup: %f\n", seq_time / par_time);
    
    return 0;
} 