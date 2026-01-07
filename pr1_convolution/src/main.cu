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

// kernels

extern "C" __global__
void erosion(
    float* img, float* ker, float* out,
    int img_H, int img_W, 
    int ker_H, int ker_W,
    int out_H, int out_W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= out_W || idy >= out_H) return;

    float min_val = FLT_MAX;

    for (int ki = 0; ki < ker_H; ++ki) {
        
        int img_row_offset = (idy + ki) * img_W;
        int ker_row_offset = ki * ker_W;

        for (int kj = 0; kj < ker_W; ++kj) {
            
            float pixel = img[img_row_offset + (idx + kj)];
            float k_val = ker[ker_row_offset + kj];

            min_val = fminf(min_val, pixel - k_val);
        }
    }

    out[idy * out_W + idx] = min_val;
}


/*
For dilation replace the following lines in the erosion code:
    float max_val = -FLT_MAX;
    max_val = fmaxf(max_val, pixel + k_val);
    out[idy * out_W + idx] = max_val;
*/



// utility

static inline float frand() {
    return (float)rand() / (float)RAND_MAX;
}


int main() {

    srand((unsigned)time(nullptr));

    // dimensions
    const int img_H = 1024;
    const int img_W = 1024;
    const int ker_H = 200;
    const int ker_W = 200;
    const int out_H = img_H - ker_H + 1;
    const int out_W = img_W - ker_W + 1;

    const size_t img_bytes = img_H * img_W * sizeof(float);
    const size_t ker_bytes = ker_H * ker_W * sizeof(float);
    const size_t out_bytes = out_H * out_W * sizeof(float);

    // host allocations
    float* h_img = (float*)malloc(img_bytes);
    float* h_ker = (float*)malloc(ker_bytes);
    float* h_out_erosion = (float*)malloc(out_bytes);

    // fill with random data
    for (int i = 0; i < img_H * img_W; ++i)
        h_img[i] = frand();

    for (int i = 0; i < ker_H * ker_W; ++i)
        h_ker[i] = frand();

    // device allocations
    float *d_img, *d_ker, *d_out;
    cudaMalloc(&d_img, img_bytes);
    cudaMalloc(&d_ker, ker_bytes);
    cudaMalloc(&d_out, out_bytes);

    cudaMemcpy(d_img, h_img, img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ker, h_ker, ker_bytes, cudaMemcpyHostToDevice);

    // launch configuration
    dim3 block(16, 16);
    dim3 grid((out_W + block.x - 1) / block.x,
              (out_H + block.y - 1) / block.y);
              
    // erosion
    auto start = std::chrono::high_resolution_clock::now();
    CHECK(cudaMalloc(&d_img, img_bytes));
    CHECK(cudaMemcpy(d_img, h_img, img_bytes, cudaMemcpyHostToDevice));
    erosion<<<grid, block>>>(
        d_img, d_ker, d_out,
        img_H, img_W,
        ker_H, ker_W,
        out_H, out_W
    );
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_out_erosion, d_out, out_bytes, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // cleanup
    cudaFree(d_img);
    cudaFree(d_ker);
    cudaFree(d_out);

    free(h_img);
    free(h_ker);
    free(h_out_erosion);
    
    auto duration_us = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Done in " << duration_us.count() << " ms.\n";

    return 0;
}
