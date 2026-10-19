# Architecture (draft – finalize in design phase, weeks 3–5)

## Data flow

```
[ESP32 + sensor] --TRIG--> [HAL: serial] --> [Core pipeline] <-- [HAL: camera]
                                                  |
                         [Preprocessor: CPU or CUDA]  (src/cpu, src/cuda)
                                                  |
                         [Classifier: CV baseline or ONNX model]  (ml/export)
                                                  |
              [Dashboard: ImGui/OpenGL] <--- result ---> [Logger: SQLite]
                                                  |
[ESP32 + servo] <--PASS/REJECT-- [HAL: serial] ---+
```

## Threads (proposed)

1. **Capture thread** – waits for a trigger, grabs a frame, pushes it to a queue.
2. **Processing thread** – preprocess + classify, pushes results.
3. **UI thread** – renders the dashboard at display rate; never blocks on processing.
4. **I/O** – serial reads/writes and SQLite writes, off the UI thread.

## Swappable stages

Every stage implements an interface in `include/aip/pipeline.hpp`, so we can swap CPU vs. CUDA preprocessing for benchmarks and a fake camera for tests.

## Open decisions

- [ ] Model runtime in C++: ONNX Runtime vs. OpenCV dnn vs. TensorRT
- [ ] Camera access: OpenCV VideoCapture vs. V4L2 directly
- [ ] Serial library choice
- [ ] Jetson board model (affects CUDA version and compiler)
