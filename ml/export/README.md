# Model export: ONNX -> TensorRT -> C++

The path from a trained model to a PASS/FAIL verdict in the C++ app.

```
ml/training/train_anomaly.py   train + evaluate + export ONNX     (desktop GPU)
ml/export/verify_onnx.py       check ONNX, build/check TensorRT   (desktop GPU)
trtexec                        build the engine on the Jetson     (on the board)
aip_trt_infer / TrtClassifier  run the engine from C++
```

## 1. Train and export

```bash
python ml/training/train_anomaly.py --data-root /path/to/mvtec_anomaly_detection --category bottle
```

Writes `ml/runs/patchcore/bottle/` (git-ignored): checkpoint, `weights/onnx/model.onnx` (+ `.onnx.data`)
and `run_info.json` with the metrics for the experiment log. `--backbone resnet18` gives a much
smaller model, which is worth trying for the Jetson.

## 2. Verify

```bash
python ml/export/verify_onnx.py --onnx ml/runs/patchcore/bottle/weights/onnx/model.onnx \
    --data-root /path/to/mvtec_anomaly_detection --category bottle --tensorrt --save-ppm ml/runs/ppm
```

Reports AUROC and accuracy for ONNX Runtime and TensorRT, how far TensorRT's scores are from ONNX
Runtime's, and saves a few test images as `.ppm` for the C++ tool. `--tensorrt` needs the `tensorrt`
Python package (desktop only).

## Model contract (fill into docs/interfaces.md once the teams agree)

| | |
| --- | --- |
| Input | `input`, 1x3x256x256 float32, RGB, 0-1, NCHW. Resize/normalization to ImageNet happens inside the model. |
| Outputs | `pred_score` [1] float32 (normalized, 0.5 = threshold), `pred_label` [1] bool, `anomaly_map` / `pred_mask` 1x1x256x256 |
| ONNX opset | 18 (anomalib 2.7's exporter; it could not convert down to 14) |

## 3. Build the engine on the target

TensorRT engines only run on the GPU model and TensorRT version that built them, so build on each
machine from the ONNX file (copy both `model.onnx` and `model.onnx.data`):

```bash
/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.fp16.engine --fp16
```

On the desktop, `verify_onnx.py --tensorrt` already wrote `model.fp16.engine` next to the ONNX file.

Jetson notes:
- **Orin Nano (JetPack 6, TensorRT 10)**: opset 18 is supported.
- **Original Nano (JetPack 4.6, TensorRT 8.2)**: supports ONNX opset 13 at most, so this export will
  not load there. It would need an older anomalib/PyTorch exporter or a different model (ML-21).

## 4. Run from C++

Configure with TensorRT available (JetPack has it system-wide; elsewhere pass `-DTENSORRT_ROOT=...`),
then:

```bash
./build/src/app/aip_trt_infer ml/runs/patchcore/bottle/weights/onnx/model.fp16.engine ml/runs/ppm/*.ppm
```

`aip::TrtClassifier` (`include/aip/trt_classifier.hpp`) is the same thing as a pipeline `Classifier`:
it takes the raw RGB frame, resizes it with `aip::to_model_input` and runs the engine. It builds
against TensorRT 8.x and 10.x.
