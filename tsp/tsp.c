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
        int minDist = INT_MAX;
        int next = -1;
        
        // Parallelized part
        #pragma omp parallel for reduction(min:minDist) reduction(max:next)
        for (int j = 0; j < n; j++) {
            if (!visited[j]) {
                int dist = calculateDistance(cities[current], cities[j]);
                if (dist < minDist) {
                    minDist = dist;
                    next = j;
                }
            }
        }
        
        // Sequential part
        totalDistance += minDist;
        current = next;
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