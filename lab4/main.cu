#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << endl; \
            exit(1); \
        } \
    } while (0)

template <int TILE_SIZE>
__global__ void matrixMulTiledKernel(const double* A, const double* B, double* C, int N) {
    __shared__ double sA[TILE_SIZE][TILE_SIZE];
    __shared__ double sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double sum = 0.0;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        if (row < N && aCol < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0;

        if (bRow < N && col < N)
            sB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main(int argc, char* argv[]) {
    int block_size = 16;
    if (argc > 1) {
        block_size = atoi(argv[1]);
    }
    if (block_size != 8 && block_size != 16 && block_size != 32) {
        cerr << "Ошибка: поддерживаются только размеры блока 8, 16 или 32!" << endl;
        return 1;
    }

    ifstream file1("input/matrix_a.txt");
    if (!file1.is_open()) {
        cerr << "Ошибка: Не удалось открыть файл input/matrix_a.txt!" << endl;
        return 1;
    }
    int m = 0;
    file1 >> m;
    vector<double> h_A(m * m);
    for (int i = 0; i < m * m; ++i) {
        file1 >> h_A[i];
    }
    file1.close();

    ifstream file2("input/matrix_b.txt");
    if (!file2.is_open()) {
        cerr << "Ошибка: Не удалось открыть файл input/matrix_b.txt!" << endl;
        return 1;
    }
    int n = 0;
    file2 >> n;
    if (m != n) {
        cerr << "Ошибка: Матрицы должны быть одинакового размера!" << endl;
        return 1;
    }
    vector<double> h_B(n * n);
    for (int i = 0; i < n * n; ++i) {
        file2 >> h_B[i];
    }
    file2.close();

    int N = m;
    size_t bytes = N * N * sizeof(double);
    vector<double> h_C(N * N, 0.0);

    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    dim3 threads(block_size, block_size);
    dim3 blocks((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);

    // Прогрев GPU
    if (block_size == 8) matrixMulTiledKernel<8><<<blocks, threads>>>(d_A, d_B, d_C, N);
    else if (block_size == 16) matrixMulTiledKernel<16><<<blocks, threads>>>(d_A, d_B, d_C, N);
    else if (block_size == 32) matrixMulTiledKernel<32><<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    if (block_size == 8) matrixMulTiledKernel<8><<<blocks, threads>>>(d_A, d_B, d_C, N);
    else if (block_size == 16) matrixMulTiledKernel<16><<<blocks, threads>>>(d_A, d_B, d_C, N);
    else if (block_size == 32) matrixMulTiledKernel<32><<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double time_sec = ms / 1000.0;

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    long long total_elements = 3LL * N * N;
    long long total_flops = 2LL * N * N * N;
    double gflops = (total_flops / 1e9) / time_sec;

    ofstream out("output/result.txt");
    if (!out.is_open()) {
        cerr << "Ошибка: Не удалось создать файл output/result.txt!" << endl;
        return 1;
    }
    out << N << endl;
    out << setprecision(15);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            out << h_C[i * N + j] << " ";
        }
        out << endl;
    }
    out.close();

    ofstream info("output/info.txt");
    if (info.is_open()) {
        info << "=== CUDA Matrix Multiplication Info ===" << endl;
        info << "Matrix size: " << N << "x" << N << endl;
        info << "Block size: " << block_size << "x" << block_size << endl;
        info << "Execution time: " << fixed << setprecision(6) << time_sec << " sec" << endl;
        info << "Task volume (Data): " << total_elements << " elements" << endl;
        info << "Task volume (Ops): " << total_flops << " FLOP" << endl;
        info << "Performance: " << fixed << setprecision(2) << gflops << " GFLOPS" << endl;
        info.close();
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}