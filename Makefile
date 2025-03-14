CC = clang
CFLAGS = -Xclang -fopenmp -lomp -O3 -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib
TARGETS = matrix_mul dijkstra tsp_solver

all: $(TARGETS)

matrix_mul: matrix_multiplication/matrix_mul.c
	$(CC) $(CFLAGS) -o $@ $<

dijkstra: graph_algorithms/dijkstra.c
	$(CC) $(CFLAGS) -o $@ $<

tsp_solver: tsp/tsp.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean 