"""Check an exported anomaly model with ONNX Runtime and, optionally, TensorRT.

Runs every image in <data-root>/<category>/test through the model and reports image-level AUROC
and accuracy at the 0.5 threshold, plus the model's input/output spec for docs/interfaces.md.

  --tensorrt        also build a TensorRT engine next to the ONNX file and check its scores
                    match ONNX Runtime (desktop TensorRT 10; on a Jetson use trtexec instead)
  --save-ppm DIR    write a few test images as .ppm files for the C++ tool aip_trt_infer

Example, from the repo root:
  python ml/export/verify_onnx.py --onnx ml/runs/patchcore/bottle/weights/onnx/model.onnx \
      --data-root /path/to/mvtec_anomaly_detection --category bottle --tensorrt
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

THRESHOLD = 0.5  # anomalib's post-processor normalizes pred_score so that 0.5 separates PASS/FAIL


def load_image(path: Path, height: int, width: int) -> np.ndarray:
    """RGB, resized with area averaging, scaled to 0-1, NCHW float32. Matches aip::to_model_input."""
    # OpenCV loads images as BGR; the model was trained on RGB. IMREAD_COLOR also turns the
    # grayscale MVTec categories (grid, screw, zipper) into 3 channels.
    rgb = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    # INTER_AREA averages every source pixel under each output pixel. It must stay the same
    # method as the C++ preprocessing, or Python and C++ scores can't be compared.
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    # HWC -> CHW (transpose), add the batch axis ([None]) -> 1x3xHxW, scale 0-255 to 0-1.
    # ascontiguousarray because the transpose is only a view and the runtimes need packed memory.
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)


def test_images(root: Path, category: str) -> list[tuple[Path, int]]:
    """Every test image with its label: 0 = good, 1 = defective (any other folder name)."""
    items = []
    for folder in sorted((root / category / "test").iterdir()):
        label = 0 if folder.name == "good" else 1
        items += [(p, label) for p in sorted(folder.glob("*.png"))]
    return items


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve, computed directly so this script needs no scikit-learn.

    AUROC equals the chance that a random defective image scores higher than a random good
    one. Compare every (defective, good) pair, count the wins, count ties as half a win.
    1.0 = the model ranks every defect above every good part; 0.5 = no better than chance.
    """
    pos, neg = scores[labels == 1], scores[labels == 0]
    # pos[:, None] vs neg[None, :] broadcasts to a len(pos) x len(neg) grid of all pairs.
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (
        pos[:, None] == neg[None, :]
    ).sum()
    return float(wins / (len(pos) * len(neg)))


def report(name: str, scores: np.ndarray, labels: np.ndarray, ms: list[float]) -> None:
    # Accuracy uses the fixed 0.5 threshold, the same rule the C++ classifier applies.
    acc = float(((scores >= THRESHOLD).astype(int) == labels).mean())
    print(
        f"{name:13s} AUROC {auroc(scores, labels):.4f} | accuracy@0.5 {acc:.4f} | "
        f"median latency {np.median(ms):.2f} ms"
    )


class TrtRunner:
    """Builds (if needed) and runs a TensorRT 10 engine, using PyTorch for GPU buffers."""

    def __init__(self, onnx_path: Path, engine_path: Path, fp16: bool) -> None:
        # Imported here so the ONNX Runtime check works on machines without TensorRT.
        import tensorrt as trt
        import torch

        self.torch = torch
        logger = trt.Logger(trt.Logger.WARNING)

        # Building an engine takes about a minute, so reuse one from an earlier run.
        # Delete the .engine file to force a rebuild (e.g. after retraining).
        if not engine_path.exists():
            builder = trt.Builder(logger)
            # 0 = default flags; TensorRT 10 always uses explicit batch dimensions.
            network = builder.create_network(0)
            parser = trt.OnnxParser(network, logger)
            # parse_from_file (not parse) so the weights in model.onnx.data next to the
            # graph are found too.
            if not parser.parse_from_file(str(onnx_path)):
                errors = "\n".join(
                    str(parser.get_error(i)) for i in range(parser.num_errors)
                )
                raise SystemExit(f"TensorRT could not parse {onnx_path}:\n{errors}")
            config = builder.create_builder_config()
            # Scratch GPU memory TensorRT may use while choosing the fastest kernels: 1 GiB.
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            # FP16 roughly halves memory and time on modern GPUs, at a small accuracy cost
            # that this script measures against ONNX Runtime below.
            if fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            print(
                f"building TensorRT engine ({'FP16' if fp16 else 'FP32'}), this takes a minute ..."
            )
            engine_path.write_bytes(builder.build_serialized_network(network, config))

        self.engine = trt.Runtime(logger).deserialize_cuda_engine(
            engine_path.read_bytes()
        )
        self.context = self.engine.create_execution_context()
        # TensorRT warns about extra syncs on the default stream
        self.stream = torch.cuda.Stream()

        # One GPU buffer per input/output tensor, allocated once and reused for every image.
        # PyTorch tensors stand in for raw cudaMalloc, so no extra CUDA package is needed.
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            # TensorRT data type -> numpy dtype -> matching torch dtype (float32, bool, ...).
            dtype = torch.from_numpy(
                np.empty(0, trt.nptype(self.engine.get_tensor_dtype(name)))
            ).dtype
            buf = torch.empty(
                tuple(self.engine.get_tensor_shape(name)), dtype=dtype, device="cuda"
            )
            # Tell TensorRT where this tensor lives in GPU memory.
            self.context.set_tensor_address(name, buf.data_ptr())
            self.buffers[name] = buf

    def __call__(self, x: np.ndarray) -> float:
        """Runs one 1x3xHxW image and returns its anomaly score."""
        # Queue the upload and the inference on our stream, then wait for both to finish.
        with self.torch.cuda.stream(self.stream):
            self.buffers["input"].copy_(self.torch.from_numpy(x))
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return float(self.buffers["pred_score"].reshape(-1)[0])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--tensorrt", action="store_true")
    ap.add_argument(
        "--fp32",
        action="store_true",
        help="Build the TensorRT engine in FP32 instead of FP16",
    )
    ap.add_argument("--save-ppm", type=Path)
    a = ap.parse_args()

    # ONNX Runtime on the CPU is the reference: it runs the exported graph as written, so any
    # difference TensorRT shows comes from TensorRT (FP16, fused kernels), not from the export.
    session = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    # Print the model contract: these names and shapes are what the C++ side relies on.
    print(f"input : {inp.name} {inp.shape} {inp.type}  (RGB, 0-1, NCHW)")
    for out in session.get_outputs():
        print(f"output: {out.name} {out.shape} {out.type}")
    _, _, height, width = inp.shape

    # Load and preprocess all images once, so the timings below measure inference only.
    items = test_images(a.data_root, a.category)
    labels = np.array([label for _, label in items])
    images = [load_image(p, height, width) for p, _ in items]

    ort_scores, ort_ms = [], []
    for x in images:
        t0 = time.perf_counter()
        # Ask only for pred_score; the anomaly map and mask outputs aren't needed here.
        ort_scores.append(
            float(session.run(["pred_score"], {inp.name: x})[0].reshape(-1)[0])
        )
        ort_ms.append((time.perf_counter() - t0) * 1000)
    ort_scores = np.array(ort_scores)
    print(f"{len(items)} test images ({labels.sum()} defective)")
    report("ONNX Runtime", ort_scores, labels, ort_ms)

    if a.tensorrt:
        # The engine is saved next to the ONNX file, e.g. model.fp16.engine.
        engine_path = a.onnx.with_suffix(".fp32.engine" if a.fp32 else ".fp16.engine")
        runner = TrtRunner(a.onnx, engine_path, fp16=not a.fp32)
        runner(images[0])  # warm-up: the first run pays one-time GPU setup costs
        trt_scores, trt_ms = [], []
        for x in images:
            t0 = time.perf_counter()
            trt_scores.append(runner(x))
            trt_ms.append((time.perf_counter() - t0) * 1000)
        trt_scores = np.array(trt_scores)
        report("TensorRT", trt_scores, labels, trt_ms)
        # The checks that matter: how far the scores drifted, and whether any image changed
        # between PASS and FAIL because of it.
        flips = int(((trt_scores >= THRESHOLD) != (ort_scores >= THRESHOLD)).sum())
        print(
            f"TensorRT vs ONNX Runtime: max score diff {np.abs(trt_scores - ort_scores).max():.4f}, "
            f"{flips} verdict(s) differ | engine: {engine_path}"
        )

    if a.save_ppm:
        # Two good and two defective images, saved at full size as binary PPM (P6). The C++
        # tool aip_trt_infer reads PPM so it needs no image library; it does its own resize.
        a.save_ppm.mkdir(parents=True, exist_ok=True)
        picks = [i for i, (_, label) in enumerate(items) if label == 0][:2]
        picks += [i for i, (_, label) in enumerate(items) if label == 1][:2]
        for i in picks:
            path, label = items[i]
            out = a.save_ppm / f"{path.parent.name}_{path.stem}.ppm"
            cv2.imwrite(str(out), cv2.imread(str(path), cv2.IMREAD_COLOR))
            print(
                f"saved {out}  (expected {'FAIL' if label else 'PASS'}, ONNX score {ort_scores[i]:.4f})"
            )


if __name__ == "__main__":
    main()
