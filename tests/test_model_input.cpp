// Unit tests for aip::to_model_input (src/core/model_input.cpp).
// Runs in CI with every build; needs no GPU or TensorRT.
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "aip/model_input.hpp"

// Same minimal check macro as test_grayscale.cpp: print the failed condition and its line,
// then exit non-zero so ctest reports the failure.
#define EXPECT(cond)                                                  \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAILED: %s (line %d)\n", #cond, __LINE__); \
            return EXIT_FAILURE;                                      \
        }                                                             \
    } while (0)

// Float results go through divisions, so compare with a small tolerance instead of ==.
static bool near(float a, float b) { return std::fabs(a - b) < 1e-5f; }

int main() {
    // 4x2 image: left half R=0, right half R=255; G is 51 everywhere; B is the row index * 255.
    // Each channel has its own pattern, so a value landing in the wrong plane or position
    // shows up as a wrong number.
    aip::Image img(4, 2, 3);
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 4; ++x) {
            std::uint8_t* p = &img.data[(y * 4 + x) * 3];
            p[0] = x < 2 ? 0 : 255;
            p[1] = 51;
            p[2] = static_cast<std::uint8_t>(y * 255);
        }
    }

    // Downscale to 2x1: each output pixel averages a 2x2 block.
    // Output layout is planar: [R0, R1, G0, G1, B0, B1].
    auto out = aip::to_model_input(img, 2, 1);
    EXPECT(out.size() == 2 * 1 * 3);
    EXPECT(near(out[0], 0.0f) && near(out[1], 1.0f));  // R plane: left block 0, right block 255
    EXPECT(near(out[2], 0.2f) && near(out[3], 0.2f));  // G plane: 51 / 255 = 0.2
    EXPECT(near(out[4], 0.5f) && near(out[5], 0.5f));  // B plane: average of rows 0 and 255

    // Same size keeps every pixel, just scaled to 0-1 and made planar.
    // Each plane holds 8 values: R at 0-7, G at 8-15, B at 16-23; inside a plane, index y * 4 + x.
    out = aip::to_model_input(img, 4, 2);
    EXPECT(near(out[3], 1.0f) && near(out[8 + 0], 0.2f) && near(out[16 + 4], 1.0f));

    // Non-integer scale: weights still sum to 1, so a flat image stays flat.
    // 7 -> 3 and 5 -> 2 cut source pixels at window edges; if the partial weights were wrong,
    // some outputs would come out darker or brighter than 128.
    aip::Image flat(7, 5, 3);
    for (auto& v : flat.data) v = 128;
    out = aip::to_model_input(flat, 3, 2);
    for (float v : out) EXPECT(near(v, 128.0f / 255.0f));

    std::puts("model input tests passed");
    return EXIT_SUCCESS;
}
