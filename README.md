# AIP – Autonomous Inspection Platform

A GPU-accelerated visual inspection station built by Code Coogs. A sensor triggers a camera, CUDA kernels preprocess the image, an ML model classifies the part as PASS or FAIL, a C++ dashboard shows the result, and an actuator sorts rejects. A core goal is measuring CPU vs. CUDA performance.

> Status: Phase 1 (setup). Most folders hold placeholders. The Jetson board is not here yet; all development targets a desktop NVIDIA GPU for now.

## Pipeline

```
sensor trigger -> capture -> CUDA preprocessing -> classification -> dashboard -> reject actuator -> log
```

## Repo layout

| Path | Owner team | Contents |
| --- | --- | --- |
| `src/app` | C++/Graphics | Main application entry point |
| `src/core` | C++/Graphics | Pipeline interfaces and orchestration |
| `src/cpu` | GPU/CUDA | CPU baseline implementations |
| `src/cuda` | GPU/CUDA | CUDA kernels |
| `src/ui` | C++/Graphics | ImGui/OpenGL dashboard (added in Phase 2) |
| `src/hal` | C++/Graphics + Hardware | Camera and serial (microcontroller) access |
| `include/aip` | All C++ | Public headers |
| `firmware/esp32` | Hardware & Robotics | Microcontroller firmware (PlatformIO) |
| `hardware` | Hardware & Robotics | CAD exports, wiring diagrams, BOM |
| `ml` | ML/Computer Vision | Training code, configs, notebooks, ONNX export |
| `data` | ML/Computer Vision | Dataset README only. Data lives in Google Drive |
| `benchmarks` | GPU/CUDA | Benchmark harness and result summaries |
| `tests` | All | Unit and integration tests |
| `docs` | All | Architecture, interfaces, setup, decisions (ADRs) |
| `deploy/jetson` | GPU/CUDA + C++ | Jetson notes (placeholder until the board arrives) |

## Build (C++)

Requirements: CMake 3.24+, a C++17 compiler. CUDA Toolkit is optional; without it the project builds CPU-only.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/src/app/aip
```

Force CPU-only: `-DAIP_ENABLE_CUDA=OFF`.

## ML environment

```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Working agreement

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: one Jira ticket per branch, branch name starts with the ticket key, every change goes through a pull request with one review, and `main` must always build.

## Docs

- [Architecture](docs/architecture.md)
- [Interfaces between teams](docs/interfaces.md)
- [Development setup](docs/setup.md)
- [Jetson notes](docs/jetson.md)
- [Decision records](docs/adr/)
