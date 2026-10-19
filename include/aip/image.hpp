#pragma once
#include <cstdint>
#include <vector>

namespace aip {

// Interleaved 8-bit image owned on the host. Agreed format lives in docs/interfaces.md.
struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;  // 1 = gray, 3 = RGB
    std::vector<std::uint8_t> data;

    Image() = default;
    Image(int w, int h, int c) : width(w), height(h), channels(c), data(static_cast<size_t>(w) * h * c) {}
    std::size_t size() const { return data.size(); }
};

}  // namespace aip
