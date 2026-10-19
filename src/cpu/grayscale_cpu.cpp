#include "aip/kernels.hpp"

#include <stdexcept>

namespace aip {

Image grayscale_cpu(const Image& rgb) {
    if (rgb.channels != 3) throw std::invalid_argument("grayscale_cpu expects 3 channels");
    Image out(rgb.width, rgb.height, 1);
    const std::size_t n = static_cast<std::size_t>(rgb.width) * rgb.height;
    for (std::size_t i = 0; i < n; ++i) {
        const auto* p = &rgb.data[i * 3];
        // ITU-R BT.601 luma, integer approximation
        out.data[i] = static_cast<std::uint8_t>((77 * p[0] + 150 * p[1] + 29 * p[2]) >> 8);
    }
    return out;
}

#ifndef AIP_HAS_CUDA
bool cuda_available() { return false; }
#endif

}  // namespace aip
