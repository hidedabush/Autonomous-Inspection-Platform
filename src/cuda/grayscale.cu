// Starter kernel: one thread per pixel. Kept to CUDA 10.2-compatible features for Jetson Nano portability.
#include <cuda_runtime.h>

#include <stdexcept>

#include "aip/kernels.hpp"

namespace aip {
namespace {

__global__ void grayscale_kernel(const unsigned char* rgb, unsigned char* gray, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const unsigned char* p = rgb + i * 3;
        gray[i] = static_cast<unsigned char>((77 * p[0] + 150 * p[1] + 29 * p[2]) >> 8);
    }
}

void check(cudaError_t err) {
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

}  // namespace

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

Image grayscale_cuda(const Image& rgb) {
    if (rgb.channels != 3) throw std::invalid_argument("grayscale_cuda expects 3 channels");
    Image out(rgb.width, rgb.height, 1);
    const int n = rgb.width * rgb.height;

    unsigned char *d_rgb = nullptr, *d_gray = nullptr;
    check(cudaMalloc(&d_rgb, rgb.size()));
    check(cudaMalloc(&d_gray, out.size()));
    check(cudaMemcpy(d_rgb, rgb.data.data(), rgb.size(), cudaMemcpyHostToDevice));

    const int block = 256;
    grayscale_kernel<<<(n + block - 1) / block, block>>>(d_rgb, d_gray, n);
    check(cudaGetLastError());

    check(cudaMemcpy(out.data.data(), d_gray, out.size(), cudaMemcpyDeviceToHost));
    cudaFree(d_rgb);
    cudaFree(d_gray);
    return out;
}

}  // namespace aip
