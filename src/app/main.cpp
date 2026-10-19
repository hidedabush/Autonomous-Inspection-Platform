// Phase 1 smoke test: runs the grayscale stage on a synthetic image, CPU and (if present) CUDA.
#include <chrono>
#include <cstdio>

#include "aip/kernels.hpp"
#include "aip/pipeline.hpp"

int main() {
    using clock = std::chrono::steady_clock;
    aip::Image img(1920, 1080, 3);
    for (std::size_t i = 0; i < img.size(); ++i) img.data[i] = static_cast<std::uint8_t>(i % 251);

    auto t0 = clock::now();
    aip::Image g = aip::grayscale_cpu(img);
    auto t1 = clock::now();
    std::printf("AIP v0.1 | grayscale CPU: %.3f ms (%dx%d)\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(), g.width, g.height);

#ifdef AIP_HAS_CUDA
    if (aip::cuda_available()) {
        aip::grayscale_cuda(img);  // warm-up
        t0 = clock::now();
        aip::grayscale_cuda(img);
        t1 = clock::now();
        std::printf("AIP v0.1 | grayscale CUDA (incl. transfers): %.3f ms\n",
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
    } else {
        std::printf("CUDA build, but no GPU detected.\n");
    }
#else
    std::printf("Built without CUDA.\n");
#endif
    return 0;
}
