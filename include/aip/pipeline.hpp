#pragma once
#include <string>

#include "aip/image.hpp"

namespace aip {

enum class Verdict { Pass, Fail, Unknown };

struct InspectionResult {
    Verdict verdict = Verdict::Unknown;
    std::string defect_class;  // empty when Pass
    float confidence = 0.0f;
    double latency_ms = 0.0;
};

// Each stage is swappable (CPU vs. CUDA, real vs. fake) so it can be tested and benchmarked.
class Preprocessor {
public:
    virtual ~Preprocessor() = default;
    virtual Image run(const Image& input) = 0;
    virtual std::string name() const = 0;
};

class Classifier {
public:
    virtual ~Classifier() = default;
    virtual InspectionResult classify(const Image& preprocessed) = 0;
};

const char* to_string(Verdict v);

}  // namespace aip
