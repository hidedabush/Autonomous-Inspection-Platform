# ML / Computer Vision

| Folder | Contents |
| --- | --- |
| `notebooks/` | Exploration (clear outputs before committing) |
| `training/` | Training and evaluation scripts |
| `configs/` | Model and dataset configs |
| `export/` | ONNX export scripts |

Plan: traditional CV baseline -> PatchCore (anomalib) on MVTec AD -> fine-tune on AIP custom dataset -> export ONNX.
Every reported result records dataset version, split and metrics in the experiment log (Drive › ML-CV › Models & Experiments).
