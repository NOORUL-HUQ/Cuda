#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix addition: C = A + B
__global__ void matrixAdd(int *A, int *B, int *C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

void printMatrix(int *matrix, int rows, int cols, const char* name) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    int rows, cols;

    // 1. Take input for matrix dimensions from the user
    printf("Enter the number of rows: ");
    scanf("%d", &rows);
    printf("Enter the number of columns: ");
    scanf("%d", &cols);

    int size = rows * cols;
    int nbytes = size * sizeof(int);

    // 2. Allocate memory on the host (CPU)
    int *h_A = (int *)malloc(nbytes);
    int *h_B = (int *)malloc(nbytes);
    int *h_C = (int *)malloc(nbytes);

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Host memory allocation failed\n");
        return -1;
    }

    // 3. Take input for matrix A from the user
    printf("Enter elements of Matrix A:\n");
    for (int i = 0; i < size; i++) {
        scanf("%d", &h_A[i]);
    }

    // 4. Take input for matrix B from the user
    printf("Enter elements of Matrix B:\n");
    for (int i = 0; i < size; i++) {
        scanf("%d", &h_B[i]);
    }

    // 5. Print the input matrices
    printMatrix(h_A, rows, cols, "A (Input)");
    printMatrix(h_B, rows, cols, "B (Input)");

    // 6. Allocate memory on the device (GPU)
    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, nbytes);
    cudaMalloc((void **)&d_B, nbytes);
    cudaMalloc((void **)&d_C, nbytes);

    // Check for CUDA errors (optional but recommended for robust code)
    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "CUDA memory allocation failed\n");
        return -1;
    }

    // 7. Copy data from host to device
    cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nbytes, cudaMemcpyHostToDevice);

    // 8. Define block and grid dimensions
    // A single 1D grid is used, which is simpler for matrix addition as long as the total size is within limits.
    // For large matrices, a 2D grid/block setup is more efficient (as suggested in NVIDIA developer forums).
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // 9. Launch the CUDA kernel
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);

    // 10. Wait for the GPU to finish
    cudaDeviceSynchronize();

    // 11. Copy the result from device to host
    cudaMemcpy(h_C, d_C, nbytes, cudaMemcpyDeviceToHost);

    // 12. Print the output matrix
    printMatrix(h_C, rows, cols, "C (Output)");

    // 13. Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 14. Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

