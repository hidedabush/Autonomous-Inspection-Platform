#include <cstdio>
#include <cstdlib>

#include "aip/kernels.hpp"

#define EXPECT(cond)                                                  \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAILED: %s (line %d)\n", #cond, __LINE__); \
            return EXIT_FAILURE;                                      \
        }                                                             \
    } while (0)

int main() {
    aip::Image img(4, 2, 3);
    for (auto& v : img.data) v = 255;
    aip::Image g = aip::grayscale_cpu(img);
    EXPECT(g.channels == 1);
    EXPECT(g.size() == 8);
    EXPECT(g.data[0] == 255);

#ifdef AIP_HAS_CUDA
    if (aip::cuda_available()) {
        aip::Image gc = aip::grayscale_cuda(img);
        EXPECT(gc.data == g.data);  // CPU and CUDA must agree
    }
#endif
    std::puts("grayscale tests passed");
    return EXIT_SUCCESS;
}
