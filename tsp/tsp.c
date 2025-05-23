#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <limits.h>

#define N 15  // Number of cities
#define MAX_THREADS 8

// Structure to hold city coordinates
typedef struct {
    int x;
    int y;
} City;

// Calculate distance between two cities
int calculateDistance(City a, City b) {
    return abs(a.x - b.x) + abs(a.y - b.y);
}

// Generate random cities
void generateCities(City cities[], int n) {
    for (int i = 0; i < n; i++) {
        cities[i].x = rand() % 100;
        cities[i].y = rand() % 100;
    }
}

// Sequential TSP implementation using nearest neighbor
int sequentialTSP(City cities[], int n) {
    int visited[N] = {0};
    int current = 0;
    int totalDistance = 0;
    visited[0] = 1;
    
    for (int i = 1; i < n; i++) {
        int minDist = INT_MAX;
        int next = -1;
        
        for (int j = 0; j < n; j++) {
            if (!visited[j]) {
                int dist = calculateDistance(cities[current], cities[j]);
                if (dist < minDist) {
                    minDist = dist;
                    next = j;
                }
            }
        }
        
        totalDistance += minDist;
        current = next;
        visited[current] = 1;
    }
    
    // Return to starting city
    totalDistance += calculateDistance(cities[current], cities[0]);
    return totalDistance;
}

// Parallel TSP implementation with OpenMP
int parallelTSP(City cities[], int n) {
    int visited[N] = {0};
    int current = 0;
    int totalDistance = 0;
    visited[0] = 1;
    
    for (int i = 1; i < n; i++) {
        int global_minDist_for_step = INT_MAX;
        int next_city_for_step = -1;
        
        // Parallelized part to find the minimum distance to an unvisited city
        // Each thread will find its local minimum for 'dist_candidate',
        // and then these local minimums are reduced to find the overall 'global_minDist_for_step'.
        #pragma omp parallel 
        {
            int local_min_dist = INT_MAX;
            #pragma omp for nowait // nowait can be used if the next step doesn't depend on all threads finishing this exact loop
            for (int j = 0; j < n; j++) {
                if (!visited[j]) {
                    int dist = calculateDistance(cities[current], cities[j]);
                    if (dist < local_min_dist) {
                        local_min_dist = dist;
                    }
                }
            }
            // Reduce all local_min_dist to global_minDist_for_step
            #pragma omp critical
            {
                if (local_min_dist < global_minDist_for_step) {
                    global_minDist_for_step = local_min_dist;
                }
            }
        } // End of parallel region

        // Sequentially find the first city 'j' that matches this global_minDist_for_step
        // This ensures a deterministic tie-break (e.g., smallest index 'j')
        if (global_minDist_for_step == INT_MAX && n > 1 && i < n) {
            // This case implies no unvisited city was found, which shouldn't happen
            // in a connected graph if i < n. Could add error or break.
            // For safety, if it happens, break to avoid infinite loop or bad access.
            break;
        }

        for (int j = 0; j < n; j++) {
            if (!visited[j]) {
                if (calculateDistance(cities[current], cities[j]) == global_minDist_for_step) {
                    next_city_for_step = j;
                    break; // Found the first one, break for smallest index tie-breaking
                }
            }
        }
        
        if (next_city_for_step == -1) {
            // All cities visited or no path found.
            // If i < n, this implies an issue or isolated unvisited cities.
            break; 
        }

        // Sequential part
        totalDistance += global_minDist_for_step;
        current = next_city_for_step;
        visited[current] = 1;
    }
    
    // Return to starting city
    totalDistance += calculateDistance(cities[current], cities[0]);
    return totalDistance;
}

int main() {
    City cities[N];
    
    // Generate random cities
    srand(time(NULL));
    generateCities(cities, N);
    
    // Run sequential version first
    double start_time = omp_get_wtime();
    int seq_distance = sequentialTSP(cities, N);
    double seq_time = omp_get_wtime() - start_time;
    printf("Sequential time: %f seconds\n", seq_time);
    
    // Test with different numbers of threads
    printf("\nThread count, Time(s), Speedup\n");
    for (int threads = 1; threads <= MAX_THREADS; threads *= 2) {
        omp_set_num_threads(threads);
        
        start_time = omp_get_wtime();
        int par_distance = parallelTSP(cities, N);
        double par_time = omp_get_wtime() - start_time;
        
        double speedup = seq_time / par_time;
        printf("%d, %f, %f\n", threads, par_time, speedup);
    }
    
    return 0;
}