#include "aip/trt_classifier.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "aip/model_input.hpp"

// TensorRT 8.5 added named I/O tensors and 10.0 removed the old binding-index API.
// The original Jetson Nano is stuck on TensorRT 8.2, so both paths are kept.
// NV_TENSORRT_MAJOR / NV_TENSORRT_MINOR come from NvInferVersion.h (included by NvInfer.h),
// so the right path is picked at compile time on each machine.
#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 5)
#define AIP_TRT_NAMED_TENSORS 1
#else
#define AIP_TRT_NAMED_TENSORS 0
#endif

namespace aip {
namespace {

// TensorRT reports problems through a logger the application provides. Warnings and errors
// go to stderr; info and verbose messages are dropped (lower severity = more serious).
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::fprintf(stderr, "[TensorRT] %s\n", msg);
    }
};

// Turns a failed CUDA call into an exception with CUDA's own error text.
void check(cudaError_t err) {
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

// Number of elements in a tensor of shape `d`, e.g. 1x3x256x256 -> 196608.
// A dimension of -1 means "decided at run time" (a dynamic shape); this class only supports
// fixed shapes, which is why train_anomaly.py exports with dynamic_shapes=None.
std::size_t volume(const nvinfer1::Dims& d) {
    std::size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] < 0) throw std::runtime_error("engine has a dynamic shape; export ONNX with a fixed input size");
        v *= static_cast<std::size_t>(d.d[i]);
    }
    return v;
}

// Bytes per element, used to size each GPU buffer. The anomalib export has float32 outputs
// plus bool masks; the rest are here so other models don't fail on allocation.
std::size_t element_size(nvinfer1::DataType t) {
    using nvinfer1::DataType;
    switch (t) {
        case DataType::kFLOAT:
        case DataType::kINT32: return 4;
        case DataType::kHALF: return 2;
        case DataType::kINT8:
        case DataType::kBOOL: return 1;
#if NV_TENSORRT_MAJOR >= 10
        // Types that only exist from TensorRT 10 on; naming them on 8.x would not compile.
        case DataType::kINT64: return 8;
        case DataType::kBF16: return 2;
        case DataType::kUINT8: return 1;
#endif
        default: throw std::runtime_error("unsupported TensorRT data type in engine I/O");
    }
}

// What we know about one engine input or output, plus its GPU buffer.
struct Tensor {
    std::string name;  // e.g. "input", "pred_score", "anomaly_map"
    bool is_input = false;
    nvinfer1::Dims dims{};
    nvinfer1::DataType type = nvinfer1::DataType::kFLOAT;
    void* device = nullptr;  // GPU memory from cudaMalloc, freed in ~Impl
};

}  // namespace

struct TrtClassifier::Impl {
    // Declaration order matters: members are destroyed bottom-up, context before engine before runtime.
    // The logger is first because the runtime keeps using it until the runtime itself is gone.
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;           // turns the .engine bytes into an engine
    std::unique_ptr<nvinfer1::ICudaEngine> engine;         // the optimized model (read-only)
    std::unique_ptr<nvinfer1::IExecutionContext> context;  // per-run state; one per thread
    std::vector<Tensor> tensors;  // every input/output, sized once in the constructor
    std::vector<void*> bindings;  // TensorRT < 8.5 only
    cudaStream_t stream = nullptr;  // our own queue of GPU work (TensorRT warns on the default one)
    const Tensor* input = nullptr;  // points into `tensors`; safe because it is never resized again
    const Tensor* score = nullptr;  // the "pred_score" output
    int width = 0;   // model input size, read from the engine
    int height = 0;
    float threshold = 0.5f;

    ~Impl() {
        // GPU memory and the stream aren't owned by TensorRT, so free them here.
        // This runs before the unique_ptr members above are destroyed.
        for (Tensor& t : tensors) cudaFree(t.device);
        if (stream) cudaStreamDestroy(stream);
    }
};

TrtClassifier::TrtClassifier(const std::string& engine_path, float threshold) : impl_(new Impl) {
    Impl& m = *impl_;
    m.threshold = threshold;

    // 1. Read the whole .engine file (the serialized, already-optimized model) into memory.
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open engine file: " + engine_path);
    std::vector<char> blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // 2. Deserialize it. This fails if the engine was built for another GPU model or another
    //    TensorRT version, which is the most common mistake, so the error message says so.
    m.runtime.reset(nvinfer1::createInferRuntime(m.logger));
    if (!m.runtime) throw std::runtime_error("createInferRuntime failed");
    m.engine.reset(m.runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!m.engine) {
        throw std::runtime_error("cannot load " + engine_path +
                                 " (engines only run on the GPU and TensorRT version that built them)");
    }
    m.context.reset(m.engine->createExecutionContext());
    if (!m.context) throw std::runtime_error("createExecutionContext failed");

    // 3. List every input and output, record its shape and type, and give it a GPU buffer.
    //    The same buffers are reused for every frame, so classify() never allocates.
#if AIP_TRT_NAMED_TENSORS
    const int count = m.engine->getNbIOTensors();
#else
    const int count = m.engine->getNbBindings();
#endif
    m.tensors.resize(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        Tensor& t = m.tensors[static_cast<std::size_t>(i)];
#if AIP_TRT_NAMED_TENSORS
        // TensorRT 8.5+: tensors are looked up by name.
        t.name = m.engine->getIOTensorName(i);
        t.is_input = m.engine->getTensorIOMode(t.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
        t.dims = m.engine->getTensorShape(t.name.c_str());
        t.type = m.engine->getTensorDataType(t.name.c_str());
#else
        // TensorRT < 8.5: tensors are "bindings" looked up by index.
        t.name = m.engine->getBindingName(i);
        t.is_input = m.engine->bindingIsInput(i);
        t.dims = m.engine->getBindingDimensions(i);
        t.type = m.engine->getBindingDataType(i);
#endif
        check(cudaMalloc(&t.device, volume(t.dims) * element_size(t.type)));
        // Tell TensorRT where this tensor's memory is: by name on 8.5+, or as the i-th entry
        // of a pointer array passed to every enqueueV2 call on older versions.
#if AIP_TRT_NAMED_TENSORS
        m.context->setTensorAddress(t.name.c_str(), t.device);
#else
        m.bindings.push_back(t.device);
#endif
        if (t.is_input) {
            if (m.input) throw std::runtime_error("expected an engine with exactly one input");
            m.input = &t;
        } else if (t.name == "pred_score") {
            m.score = &t;
        }
        // Other outputs (pred_label, anomaly_map, pred_mask) still get buffers because
        // TensorRT writes them on every run; they are just not read yet (ML-20 will show the map).
    }

    // 4. Check the engine is the shape this class was written for, so a wrong model fails
    //    here with a clear message instead of producing nonsense scores later.
    if (!m.input) throw std::runtime_error("engine has no input tensor");
    if (!m.score) throw std::runtime_error("engine has no `pred_score` output");
    const nvinfer1::Dims& in = m.input->dims;
    if (in.nbDims != 4 || in.d[0] != 1 || in.d[1] != 3 || m.input->type != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("expected a 1x3xHxW float32 input");
    }
    if (m.score->type != nvinfer1::DataType::kFLOAT) throw std::runtime_error("expected a float32 `pred_score`");
    m.height = in.d[2];  // NCHW: d[2] is the height, d[3] the width
    m.width = in.d[3];
    check(cudaStreamCreate(&m.stream));
}

// Defined here, not in the header, because ~unique_ptr<Impl> needs Impl's full definition.
TrtClassifier::~TrtClassifier() = default;

int TrtClassifier::input_width() const { return impl_->width; }
int TrtClassifier::input_height() const { return impl_->height; }

InspectionResult TrtClassifier::classify(const Image& rgb) {
    using clock = std::chrono::steady_clock;
    Impl& m = *impl_;
    const auto t0 = clock::now();

    // Resize and convert on the CPU (a CUDA version is GPU-15), then queue the upload.
    const std::vector<float> input = to_model_input(rgb, m.width, m.height);
    check(cudaMemcpyAsync(m.input->device, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice,
                          m.stream));
    // Queue the inference on the same stream, so it starts only after the upload finishes.
#if AIP_TRT_NAMED_TENSORS
    const bool ok = m.context->enqueueV3(m.stream);
#else
    const bool ok = m.context->enqueueV2(m.bindings.data(), m.stream, nullptr);
#endif
    if (!ok) throw std::runtime_error("TensorRT inference failed");
    // Only the single score float comes back to the CPU; the larger outputs stay on the GPU.
    float score = 0.0f;
    check(cudaMemcpyAsync(&score, m.score->device, sizeof(float), cudaMemcpyDeviceToHost, m.stream));
    // Everything above was only queued; wait here until `score` actually holds the result.
    check(cudaStreamSynchronize(m.stream));

    InspectionResult r;
    r.verdict = score >= m.threshold ? Verdict::Fail : Verdict::Pass;
    r.defect_class = r.verdict == Verdict::Fail ? "anomaly" : "";  // defect categories come with ML-18
    // Normalized anomaly score, not a calibrated probability (calibration is ML-19).
    // Reported as confidence in the verdict: a score of 0.9 is a FAIL at 0.9, a score of 0.2 is
    // a PASS at 0.8. The raw score can be recovered from the verdict and the confidence.
    r.confidence = r.verdict == Verdict::Fail ? score : 1.0f - score;
    r.latency_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    return r;
}

}  // namespace aip
