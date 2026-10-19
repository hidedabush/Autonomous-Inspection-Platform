#pragma once
#include "aip/image.hpp"

namespace aip {

// CPU baseline (always available).
Image grayscale_cpu(const Image& rgb);

#ifdef AIP_HAS_CUDA
// CUDA version. Returns the result on the host; transfer time is included.
Image grayscale_cuda(const Image& rgb);
#endif

bool cuda_available();

}  // namespace aip
