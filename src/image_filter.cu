/* 
It applies a 5×5 blur (box filter) to a single-channel 2D image (H×W floats).
The filter is stored in constant memory (__constant__ float d_kernel[25]) so all threads can broadcast the same small read-only kernel efficiently.
*/

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

// Simple 5x5 blur kernel (normalized)
__constant__ float d_kernel[25]; // 5x5 kernel, the filter stored in constant memory. 

template<int TILE, int K>
__global__ void conv2d_shared(const float* __restrict__ in, float* __restrict__ out, 
                              int H, int W) {
    constexpr int R = K / 2;                // radius
    // Shared tile with apron (halo)
    __shared__ float tile[TILE + 2*R][TILE + 2*R]; // shared memory tile including halo

    int ox = blockIdx.x * TILE + threadIdx.x - R; // output x coord
    int oy = blockIdx.y * TILE + threadIdx.y - R; // output y coord

    // Load with clamping at borders
    int x = max(0, min(W - 1, ox)); // clamp x coordinate
    int y = max(0, min(H - 1, oy)); // clamp y coordinate
    tile[threadIdx.y][threadIdx.x] = in[y * W + x]; // load input pixel into shared memory

    __syncthreads(); // Ensure all loads are done

    // Only the inner TILE×TILE threads compute output
    if (threadIdx.x >= R && threadIdx.x < TILE + R && // valid output region
        threadIdx.y >= R && threadIdx.y < TILE + R) { // valid output region
        int outx = blockIdx.x * TILE + (threadIdx.x - R); // output x coord
        int outy = blockIdx.y * TILE + (threadIdx.y - R); // output y coord
        if (outx < W && outy < H) { // 
            float sum = 0.f; // accumulator
            #pragma unroll // unroll kernel loops
            for (int ky = 0; ky < K; ++ky) { // kernel application
              #pragma unroll // unroll kernel loops
              for (int kx = 0; kx < K; ++kx) { // kernel application
                sum += tile[threadIdx.y - R + ky][threadIdx.x - R + kx] *
                       d_kernel[ky * K + kx]; // apply kernel
              }
            }
            out[outy * W + outx] = sum; //
        }
    }
}

int main() {
    const int H = 512, W = 512; // image dimensions
    const int N = H * W; // total pixels

    // Host data (impulse image: center pixel = 1)
    std::vector<float> h_in(N, 0.f), h_out(N, 0.f); // host input and output
    h_in[(H/2) * W + (W/2)] = 1.f; // impulse at center

    // 5x5 normalized box filter
    float h_kernel[25]; // host kernel
    for (int i = 0; i < 25; ++i) h_kernel[i] = 1.f / 25.f;// initialize box filter
    CHECK_CUDA(cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(h_kernel))); // copy kernel to constant memory

    // Device buffers
    float *d_in = nullptr, *d_out = nullptr; // device input and output
    CHECK_CUDA(cudaMalloc(&d_in,  N * sizeof(float))); 
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float))); 
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice)); // copy input to device

    // Launch config
    constexpr int K = 5; // kernel size
    constexpr int TILE = 16; // tile size
    dim3 block(TILE + K - 1, TILE + K - 1);     // include halo loaders
    dim3 grid((W + TILE - 1)/TILE, (H + TILE - 1)/TILE); // enough blocks to cover image

    conv2d_shared<TILE, K><<<grid, block>>>(d_in, d_out, H, W); // launch kernel
    CHECK_CUDA(cudaPeekAtLastError()); // check launch
    CHECK_CUDA(cudaDeviceSynchronize()); // wait for completion

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost)); // copy output back to host

    // Quick sanity: center should smear to a 5x5 patch ≈ 1/25
    float center = h_out[(H/2) * W + (W/2)]; // center pixel value
    printf("Center value after blur: %.6f (expected ~0.04)\n", center);
    printf("Success!\n");

    cudaFree(d_in); cudaFree(d_out); // free device memory
    return 0;
}
