#include "aip/pipeline.hpp"

namespace aip {

const char* to_string(Verdict v) {
    switch (v) {
        case Verdict::Pass: return "PASS";
        case Verdict::Fail: return "FAIL";
        default: return "UNKNOWN";
    }
}

}  // namespace aip
