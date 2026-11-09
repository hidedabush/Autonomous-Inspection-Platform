#pragma once
#include <vector>

#include "aip/image.hpp"

namespace aip {

// Converts an 8-bit RGB image to the model input: resized to width x height with area averaging
// (same as cv2.INTER_AREA in ml/export/verify_onnx.py), scaled to 0-1, NCHW float32.
// CPU baseline for the CUDA version (GPU-15).
//
// Input:  `rgb` must have 3 channels, interleaved RGBRGB... (HWC), values 0-255, any size.
// Output: 3 * width * height floats in planar order (NCHW with N = 1): first every R value
//         row by row, then every G value, then every B value, each scaled to 0-1.
//         Element (c, y, x) is at index c * width * height + y * width + x.
// Throws std::invalid_argument for a non-RGB image or a non-positive target size.
//
// The exported model normalizes (ImageNet mean/std) inside its own graph, so this function
// only resizes and scales; it must not normalize again.
std::vector<float> to_model_input(const Image& rgb, int width, int height);

}  // namespace aip
