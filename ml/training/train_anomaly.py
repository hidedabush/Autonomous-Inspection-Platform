"""Train an anomaly-detection model on one MVTec AD category, evaluate it and export ONNX.

The exported model contains anomalib's pre- and post-processing, so its input is an RGB image
scaled to 0-1 (NCHW float32) and its `pred_score` output is normalized so that 0.5 is the
PASS/FAIL threshold. See ml/export/README.md for the next steps (verify, TensorRT).

Example, from the repo root:
  python ml/training/train_anomaly.py --data-root /path/to/mvtec_anomaly_detection --category bottle
"""

import argparse
import json
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore

# Models selectable with --model. Both learn only from defect-free ("good") images, so no
# defect labels are needed:
# - PatchCore stores features of normal image patches in a memory bank and scores a new image
#   by how far its patches are from the closest stored ones. Best accuracy on MVTec, but the
#   memory bank makes the model large (about 230 MB with the default backbone).
# - PaDiM fits a Gaussian per patch position instead. Smaller and faster, usually less accurate.
MODELS = {"patchcore": Patchcore, "padim": Padim}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # The MVTec root, i.e. the folder holding one sub-folder per category. Each category has
    # train/good (training images) and test/<defect type> (evaluation images, "good" included).
    ap.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Folder that contains bottle/, cable/, ...",
    )
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--model", choices=sorted(MODELS), default="patchcore")
    # The backbone is the pretrained CNN that turns image patches into feature vectors.
    # PatchCore defaults to wide_resnet50_2; resnet18 is far smaller, which matters on a Jetson.
    ap.add_argument(
        "--backbone",
        help="Override the default backbone, e.g. resnet18 for a lighter Jetson model",
    )
    # The exported model takes square images of this size. The C++ side resizes camera frames
    # to it with aip::to_model_input, so both sides must agree on it.
    ap.add_argument(
        "--image-size", type=int, default=256, help="Square ONNX input size"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("ml/runs"),
        help="Run folder root (git-ignored)",
    )
    # Background processes that load images while the GPU works.
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    # Only pass a backbone when one was given, so each model keeps its own default otherwise.
    model_kwargs = {"backbone": a.backbone} if a.backbone else {}
    model = MODELS[a.model](**model_kwargs)

    # anomalib's MVTec loader: train/good for training, test/* for evaluation (labels and
    # ground-truth masks come from the folder names and the ground_truth/ folder).
    datamodule = MVTecAD(
        root=a.data_root,
        category=a.category,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=a.workers,
    )

    # Everything from this run lands in ml/runs/<model>/<category>/: Lightning checkpoints,
    # the exported ONNX file and run_info.json. The runs/ folder is git-ignored.
    run_dir = a.out / a.model / a.category
    engine = Engine(default_root_dir=run_dir)
    # For PatchCore, "fit" is a single pass over the good images that fills the memory bank;
    # there is no gradient training. anomalib also sets the PASS/FAIL threshold here.
    engine.fit(model=model, datamodule=datamodule)
    # Scores the test images and returns metrics such as image_AUROC and pixel_AUROC.
    metrics = engine.test(model=model, datamodule=datamodule)[0]

    # Fully static shapes (batch 1): anomalib otherwise traces a dynamic batch with an example
    # batch of 2, which the exporter can bake in. TensorRT is simplest with fixed shapes anyway.
    # The exported graph includes anomalib's resize + ImageNet normalization on the way in and
    # its score normalization on the way out, so callers only feed RGB pixels scaled to 0-1.
    size = (a.image_size, a.image_size)
    onnx_path = engine.export(
        model=model,
        export_type=ExportType.ONNX,
        export_root=run_dir,
        input_size=size,
        onnx_kwargs={"dynamic_shapes": None},
    )

    # Record what produced this model, for the experiment log in Drive.
    info = {
        "dataset": "MVTec AD",
        "category": a.category,
        "model": a.model,
        "backbone": a.backbone or "default",
        "input_size": list(size),
        # Metric values come back as tensors; round them to plain floats for JSON.
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "onnx": str(onnx_path),
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
