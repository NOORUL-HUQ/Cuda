#include <iostream>
#include <cuda_runtime.h>

// CUDA Kernel for Matrix Addition
__global__ void matrixAdd(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * N + col; // Convert 2D to 1D index

    if (row < N && col < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Helper to print matrix
void printMatrix(const float *M, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", M[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    int N = 3; // Matrix size N x N
    size_t size = N * N * sizeof(float);

    // Host memory allocation
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize input matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    printf("Input Matrix A:\n");
    printMatrix(h_A, N);
    printf("Input Matrix B:\n");
    printMatrix(h_B, N);

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(N, N);
    dim3 numBlocks(1, 1);
    matrixAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Result Matrix C (A+B):\n");
    printMatrix(h_C, N);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

