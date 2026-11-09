// Runs the TensorRT classifier on .ppm images: aip_trt_infer <model.engine> <image.ppm> [...]
// Uses PPM so it needs no image library; ml/export/verify_onnx.py --save-ppm writes test images.
//
// Output, one line per image: verdict, raw anomaly score, time for that call, file name.
// The scores should match the "ONNX score" printed by verify_onnx.py --save-ppm to within
// FP16 precision (about 0.01); a bigger gap means the preprocessing differs between the two.
#include <cstdio>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>

#include "aip/trt_classifier.hpp"

namespace {

// Reads a binary PPM (P6, 8-bit) into an RGB image.
// The format: a text header "P6 <width> <height> <maxval>" (whitespace-separated, may contain
// "# comment" lines), one whitespace byte, then width * height RGB triplets as raw bytes.
// That raw layout is exactly aip::Image's, so the pixels are read straight into it.
aip::Image load_ppm(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    // Next header token, skipping comment lines.
    auto token = [&f]() {
        std::string t;
        while (f >> t && t[0] == '#') std::getline(f, t);  // skip comments
        return t;
    };
    if (token() != "P6") throw std::runtime_error(path + " is not a binary PPM (P6)");
    const int w = std::stoi(token());
    const int h = std::stoi(token());
    if (std::stoi(token()) != 255) throw std::runtime_error(path + " is not 8-bit");
    f.get();  // the single whitespace byte before pixel data
    aip::Image img(w, h, 3);
    if (!f.read(reinterpret_cast<char*>(img.data.data()), static_cast<std::streamsize>(img.size()))) {
        throw std::runtime_error(path + " is truncated");
    }
    return img;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.engine> <image.ppm> [more.ppm ...]\n", argv[0]);
        return 2;
    }
    try {
        // Loading the engine is the slow part (it also allocates GPU memory), so do it once.
        aip::TrtClassifier classifier(argv[1]);
        std::printf("engine %s | input %dx%d\n", argv[1], classifier.input_width(), classifier.input_height());
        for (int i = 2; i < argc; ++i) {
            const aip::Image img = load_ppm(argv[i]);
            // The first run on the GPU pays one-time setup costs; time the second one.
            classifier.classify(img);  // warm-up
            const aip::InspectionResult r = classifier.classify(img);
            // TrtClassifier reports confidence in its verdict; undo that to print the raw
            // score, which is what verify_onnx.py prints for comparison.
            const float score = r.verdict == aip::Verdict::Fail ? r.confidence : 1.0f - r.confidence;
            std::printf("%-4s score %.4f | %.2f ms | %s\n", aip::to_string(r.verdict), score, r.latency_ms, argv[i]);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
