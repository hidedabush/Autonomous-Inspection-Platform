# Jetson notes (placeholder – board not yet on hand)

Develop on a desktop NVIDIA GPU until the board arrives. First task when it arrives: confirm the exact model.

| | Jetson Nano (original) | Jetson Orin Nano |
| --- | --- | --- |
| Latest JetPack | 4.6.x | 6.x |
| Ubuntu | 18.04 | 22.04 |
| CUDA | 10.2 | 12.x |
| GPU arch (CMAKE_CUDA_ARCHITECTURES) | 53 (Maxwell) | 87 (Ampere) |
| Python | 3.6 | 3.10 |
| PyTorch | 1.10-era wheels only | 2.x |

## Portability rules until then
- C++17, but expect GCC 7 on the original Nano: avoid newer library features.
- Kernels: only CUDA features available in 10.2.
- Models: train on desktop, deploy via ONNX -> TensorRT on the device.
- Jetson memory is shared between CPU and GPU: benchmark zero-copy vs. explicit copies there.

## When the board arrives
- [ ] Flash JetPack, record version here
- [ ] Run `deviceQuery`, record compute capability
- [ ] Build this repo on-device with `-DCMAKE_CUDA_ARCHITECTURES=53` (or 87)
- [ ] Run the grayscale benchmark and add results to `benchmarks/`
