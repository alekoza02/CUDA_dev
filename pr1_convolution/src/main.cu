#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <float.h>
#include <chrono>
#include <iostream>

#define CHECK(x) do { \
  cudaError_t err = x; \
  if (err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while(0)


// Hard cap limit: sizeof(kernel) < 64KB
// Formula: KCH * KCW * 4 < 64'000'000
#define KCH 100
#define KCW 100

__constant__ float c_ker[KCH * KCW];

// KERNEL
extern "C" __global__
void erosion_chunked(
    float* img,
    float* out,
    int img_H, int img_W,
    int out_H, int out_W,
    int ky_off, int kx_off)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_W || y >= out_H) return;

    int out_idx = y * out_W + x;
    float acc = out[out_idx];

    for (int ky = 0; ky < KCH; ++ky) {
        int img_y = y + ky + ky_off;
        int img_row = img_y * img_W;

        int ker_row = ky * KCW;

        for (int kx = 0; kx < KCW; ++kx) {
            float pixel = img[img_row + x + kx + kx_off];
            float kval  = c_ker[ker_row + kx];
            acc = fminf(acc, pixel - kval);
        }
    }

    out[out_idx] = acc;
}

// UTILITY
static inline float frand() {
    return (float)rand() / (float)RAND_MAX;
}

// MAIN
int main() {

    srand((unsigned)time(nullptr));

    const int img_H = 1024;
    const int img_W = 1024;
    const int ker_H = 200;
    const int ker_W = 200;
    const int out_H = img_H - ker_H + 1;
    const int out_W = img_W - ker_W + 1;

    const size_t img_bytes = img_H * img_W * sizeof(float);
    const size_t ker_bytes = ker_H * ker_W * sizeof(float);
    const size_t out_bytes = out_H * out_W * sizeof(float);

    // Host memory
    float* h_img = (float*)malloc(img_bytes);
    float* h_ker = (float*)malloc(ker_bytes);
    float* h_out = (float*)malloc(out_bytes);

    for (int i = 0; i < img_H * img_W; ++i)
        h_img[i] = frand();

    for (int i = 0; i < ker_H * ker_W; ++i)
        h_ker[i] = frand();

    for (int i = 0; i < out_H * out_W; ++i)
        h_out[i] = FLT_MAX;

    // Device memory
    float *d_img, *d_out;
    CHECK(cudaMalloc(&d_img, img_bytes));
    CHECK(cudaMalloc(&d_out, out_bytes));

    CHECK(cudaMemcpy(d_img, h_img, img_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_out, h_out, out_bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((out_W + block.x - 1) / block.x,
              (out_H + block.y - 1) / block.y);

    auto start = std::chrono::high_resolution_clock::now();

    int n_chunk_x = (ker_W + KCW - 1) / KCW;
    int n_chunk_y = (ker_H + KCH - 1) / KCH;
    
    for (int ky = 0; ky < n_chunk_y; ++ky) {
        for (int kx = 0; kx < n_chunk_x; ++kx) {

            // copy kernel chunk into constant memory
            float h_chunk[KCH * KCW];

            for (int i = 0; i < KCH; ++i)
                for (int j = 0; j < KCW; ++j)
                    h_chunk[i * KCW + j] = h_ker[(ky * KCH + i) * ker_W + (kx * KCW + j)];

            CHECK(cudaMemcpyToSymbol(c_ker, h_chunk, sizeof(h_chunk)));

            erosion_chunked<<<grid, block>>>(
                d_img, d_out,
                img_H, img_W,
                out_H, out_W,
                ky * KCH, kx * KCW
            );

            CHECK(cudaGetLastError());
        }
    }

    CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));

    std::cout << "Done in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    cudaFree(d_img);
    cudaFree(d_out);

    free(h_img);
    free(h_ker);
    free(h_out);

    return 0;
}
