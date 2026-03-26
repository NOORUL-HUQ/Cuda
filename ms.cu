#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUDA Kernel for matrix scaling
__global__ void scaleMatrix(float *matrix, float *output, int rows, int cols, float scalar) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        output[row * cols + col] = matrix[row * cols + col] * scalar;
    }
}

void printMatrix(const std::vector<float>& mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    int rows, cols;
    float scalar;
    std::cout << "Enter rows and columns: ";
    std::cin >> rows >> cols;
    std::cout << "Enter scaling factor: ";
    std::cin >> scalar;

    int size = rows * cols * sizeof(float);
    std::vector<float> h_in(rows * cols), h_out(rows * cols);

    std::cout << "Enter matrix elements:\n";
    for (int i = 0; i < rows * cols; ++i) std::cin >> h_in[i];

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    scaleMatrix<<<numBlocks, threadsPerBlock>>>(d_in, d_out, rows, cols, scalar);

    cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost);

    std::cout << "\nInput Matrix:\n";
    printMatrix(h_in, rows, cols);
    std::cout << "\nScaled Matrix:\n";
    printMatrix(h_out, rows, cols);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

