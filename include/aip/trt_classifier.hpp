#pragma once
#include <memory>
#include <string>

#include "aip/pipeline.hpp"

namespace aip {

// Runs a TensorRT engine built from the anomalib ONNX export (see ml/export/README.md).
// The engine takes one RGB 0-1 NCHW float input and has a `pred_score` output where
// scores at or above the threshold (0.5 after anomalib's normalization) mean FAIL.
//
// Usage:
//   aip::TrtClassifier classifier("model.fp16.engine");   // load once at startup (slow)
//   aip::InspectionResult r = classifier.classify(frame);  // per frame (milliseconds)
//
// The .engine file must be built on the same GPU model and TensorRT version that runs it:
// build it on each machine from the ONNX file with trtexec.
// Not thread-safe: one classify() at a time per instance (it reuses one set of GPU buffers).
class TrtClassifier : public Classifier {
public:
    // Loads and checks the engine and allocates its GPU buffers. Throws std::runtime_error with
    // a readable message if the file is missing, was built elsewhere, or has unexpected I/O.
    explicit TrtClassifier(const std::string& engine_path, float threshold = 0.5f);
    ~TrtClassifier() override;

    // Takes the raw RGB frame; resizing to the model input happens inside.
    // Fills verdict, defect_class ("anomaly" on FAIL), confidence and latency_ms
    // (latency covers resize + upload + inference + download).
    InspectionResult classify(const Image& rgb) override;

    // The model's input size, read from the engine (256 x 256 for the current export).
    int input_width() const;
    int input_height() const;

private:
    // "Pimpl": every TensorRT and CUDA type lives in the .cpp, so code that includes this
    // header compiles without TensorRT installed.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aip
