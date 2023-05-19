#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define TILE_SIZE 16

__global__ void matrixMulTiled(int *a, int *b, int *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int tileA[TILE_SIZE][TILE_SIZE];
    __shared__ int tileB[TILE_SIZE][TILE_SIZE];

    int value = 0;
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int tileRow = threadIdx.y;
        int tileCol = threadIdx.x;

        if (tileRow < TILE_SIZE && (t * TILE_SIZE + tileCol) < n) {
            tileA[tileRow][tileCol] = a[row * n + t * TILE_SIZE + tileCol];
        } else {
            tileA[tileRow][tileCol] = 0;
        }

        if (tileCol < TILE_SIZE && (t * TILE_SIZE + tileRow) < n) {
            tileB[tileRow][tileCol] = b[(t * TILE_SIZE + tileRow) * k + col];
        } else {
            tileB[tileRow][tileCol] = 0;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            value += tileA[tileRow][i] * tileB[i][tileCol];
        }

        __syncthreads();
    }

    if (row < m && col < k) {
        c[row * k + col] = value;
    }
}

void initMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = rand() % 10;
    }
}

void printMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void matrixMulSequential(int *a, int *b, int *c, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            int sum = 0;
            for (int x = 0; x < n; x++) {
                sum += a[i * n + x] * b[x * k + j];
            }
            c[i * k + j] = sum;
        }
    }
}


int main() {
    int m, n, k;
    printf("Enter the dimensions of matrices (m, n, k): ");
    scanf("%d %d %d", &m, &n, &k);

    int *a = (int *)malloc(m * n * sizeof(int));
    int *b = (int *)malloc(n * k * sizeof(int));
    int *c = (int *)malloc(m * k * sizeof(int));

    initMatrix(a, m, n);
    initMatrix(b, n, k);

    struct timeval startSeq, endSeq;
    gettimeofday(&startSeq, NULL);

    matrixMulSequential(a, b, c, m, n, k);

    gettimeofday(&endSeq, NULL);
    float elapsedTimeSeq = (endSeq.tv_sec - startSeq.tv_sec) * 1000.0f + (endSeq.tv_usec - startSeq.tv_usec) / 1000.0f;

    int *dev_a, *dev_b, *dev_c;
    cudaMalloc((void **)&dev_a, m * n * sizeof(int));
    cudaMalloc((void **)&dev_b, n * k * sizeof(int));
    cudaMalloc((void **)&dev_c, m * k * sizeof(int));


    cudaMemcpy(dev_a, a, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * k * sizeof(int), cudaMemcpyHostToDevice);


    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((k + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    matrixMulTiled<<<gridDim, blockDim>>>(dev_a, dev_b, dev_c, m, n, k);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(c, dev_c, m * k * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nResult matrix:\n");
    printMatrix(c, m, k);

    printf("\nSequential Execution Time: %.2f ms\n", elapsedTimeSeq);
    printf("Parallel Execution Time: %.2f ms\n", elapsedTime);


    float speedup = elapsedTimeSeq / elapsedTime;
    float efficiency = (speedup / (gridDim.x * gridDim.y * blockDim.x * blockDim.y)) * 100;

    printf("Speedup: %.2f\n", speedup);
    printf("Efficiency: %.2f\n", efficiency);

    printf("Number of Threads: %d\n", gridDim.x * gridDim.y * blockDim.x * blockDim.y);

    free(a);
    free(b);
    free(c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
