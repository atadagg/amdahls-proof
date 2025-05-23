#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
#include <time.h>

#define V 1000  // Number of vertices
#define INF INT_MAX
#define MAX_THREADS 8

// Function to find the vertex with minimum distance value
int minDistance(int dist[], int sptSet[]) {
    int min = INF;
    int min_index = -1; // Initialize to -1 to indicate no valid index found yet
    
    #pragma omp parallel 
    {
        int local_min = INF;
        int local_min_index = -1;
        
        #pragma omp for nowait
        for (int v = 0; v < V; v++) {
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
    
    // Run sequential version first
    double start_time = omp_get_wtime();
    sequentialDijkstra(graph, 0, dist);
    double seq_time = omp_get_wtime() - start_time;
    printf("Sequential time: %f seconds\n", seq_time);
    
    // Test with different numbers of threads
    printf("\nThread count, Time(s), Speedup\n");
    for (int threads = 1; threads <= MAX_THREADS; threads *= 2) {
        omp_set_num_threads(threads);
        
        start_time = omp_get_wtime();
        parallelDijkstra(graph, 0, dist);
        double par_time = omp_get_wtime() - start_time;
        
        double speedup = seq_time / par_time;
        printf("%d, %f, %f\n", threads, par_time, speedup);
    }
    
    return 0;
} 