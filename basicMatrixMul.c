#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BLOCK_SIZE 16

__global__ void matrixMul(int *a, int *b, int *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        int value = 0;
        for (int i = 0; i < n; ++i) {
            value += a[row * n + i] * b[i * k + col];
        }
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

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((k + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    matrixMul<<<gridDim, blockDim>>>(dev_a, dev_b, dev_c, m, n, k);

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
