#include "aip/model_input.hpp"

#include <algorithm>
#include <stdexcept>

namespace aip {
namespace {

// One source pixel's contribution to an output pixel.
struct Tap {
    int index;     // source column (or row) index
    float weight;  // share of the output pixel this source pixel covers
};

// For each output pixel, the source pixels it covers and how much of each (weights sum to 1).
//
// Area averaging treats each output pixel as a window over the source. Shrinking 900 pixels to
// 256 gives a window of 900 / 256 = 3.515625 source pixels: output pixel 0 covers source 0..3.52,
// output 1 covers 3.52..7.03, and so on. Source pixels fully inside a window count fully; the
// pixel cut by a window edge counts by the fraction inside. Dividing each overlap by the window
// width makes the weights a true average. This is what cv2.INTER_AREA does when shrinking.
//
// The same taps are used for every row (for x) or every column (for y), so they are computed
// once per axis instead of once per pixel.
std::vector<std::vector<Tap>> area_taps(int src, int dst) {
    std::vector<std::vector<Tap>> taps(static_cast<std::size_t>(dst));
    const double scale = static_cast<double>(src) / dst;  // window width in source pixels
    for (int o = 0; o < dst; ++o) {
        const double begin = o * scale;  // window [begin, end) in source coordinates
        const double end = begin + scale;
        // Visit every source pixel s whose span [s, s + 1) overlaps the window.
        for (int s = static_cast<int>(begin); s < src && s < end; ++s) {
            const double overlap = std::min(end, s + 1.0) - std::max(begin, static_cast<double>(s));
            if (overlap > 0) taps[o].push_back({s, static_cast<float>(overlap / scale)});
        }
    }
    return taps;
}

}  // namespace

std::vector<float> to_model_input(const Image& rgb, int width, int height) {
    if (rgb.channels != 3) throw std::invalid_argument("to_model_input expects 3 channels");
    if (width <= 0 || height <= 0) throw std::invalid_argument("to_model_input needs a positive size");

    const auto xs = area_taps(rgb.width, width);    // horizontal: which columns feed each output column
    const auto ys = area_taps(rgb.height, height);  // vertical: which rows feed each output row

    // The 2-D average is separable: averaging along x and then along y gives the same result as
    // averaging each 2-D box directly, at a fraction of the work for large frames.

    // Horizontal pass: rgb.height rows x width columns x 3, still interleaved.
    // Each source row shrinks to the output width; the number of rows is unchanged for now.
    std::vector<float> rows(static_cast<std::size_t>(rgb.height) * width * 3, 0.0f);
    for (int y = 0; y < rgb.height; ++y) {
        const std::uint8_t* src = &rgb.data[static_cast<std::size_t>(y) * rgb.width * 3];
        float* dst = &rows[static_cast<std::size_t>(y) * width * 3];
        for (int x = 0; x < width; ++x) {
            for (const Tap& t : xs[x]) {
                // Interleaved layout: pixel i's channel c is at i * 3 + c.
                for (int c = 0; c < 3; ++c) dst[x * 3 + c] += t.weight * src[t.index * 3 + c];
            }
        }
    }

    // Vertical pass into planar NCHW, scaled to 0-1.
    // Each output row averages the horizontally-shrunk rows its window covers, and every value
    // is written to its channel's plane: plane c starts at c * plane.
    const std::size_t plane = static_cast<std::size_t>(width) * height;
    std::vector<float> out(plane * 3, 0.0f);
    for (int y = 0; y < height; ++y) {
        for (const Tap& t : ys[y]) {
            const float* src = &rows[static_cast<std::size_t>(t.index) * width * 3];
            for (int x = 0; x < width; ++x) {
                for (int c = 0; c < 3; ++c) {
                    out[c * plane + static_cast<std::size_t>(y) * width + x] += t.weight * src[x * 3 + c];
                }
            }
        }
    }
    // Averages are still on the 0-255 scale; the model expects 0-1.
    for (float& v : out) v /= 255.0f;
    return out;
}

}  // namespace aip
