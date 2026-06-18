# Detection (YOLOv11)

Object detection on Brawl Stars frames. Input: an image. Output: detected
classes with bounding boxes and confidences.

## Setup

Python 3.11 or 3.12 (not 3.14 — no torch/ultralytics wheels yet).

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
```

The dataset (Roboflow export, YOLOv11 format) must sit in `data/dataset/`
with `train/` and `valid/` splits. The `test/` split is created automatically
on first run.

## Train

```bash
.venv/bin/python -m BROWT.detection.train          # full run
.venv/bin/python -m BROWT.detection.train --smoke  # 1-epoch sanity check
```

All settings live in [`configs/detection.yaml`](../../../configs/detection.yaml)
(model size, epochs, image size, device, ...). On first run the trainer:

1. Carves a deterministic `test` split (10% of `train`).
2. Writes a clean `data.yaml` with an absolute path root.
3. Trains, evaluates on the `test` split, and copies the best weights to
   `data/models/brawl_yolo11<size>.pt`.

Device is auto-selected: CUDA > MPS (Apple Silicon) > CPU.

Monitor:

```bash
tail -f logs/train_full.log
.venv/bin/tensorboard --logdir runs
```

Run artifacts (curves, confusion matrix, weights) land in
`runs/detect/<name>/`.

## Inference

```python
from BROWT.detection import Detector

det = Detector("data/models/brawl_yolo11n.pt")
for d in det.detect("path/to/frame.png"):
    print(d.cls_name, round(d.confidence, 2), d.center)
```

`detect()` accepts a file path or a numpy array (e.g. an `mss` capture) and
returns a list of `Detection(cls_id, cls_name, confidence, xyxy, center)`.

## Classes

10 classes: `Ball`, `Enemy`, `Friendly`, `Gem`, `Hot_Zone`, `Me`, `PP`,
`PP_Box`, `Safe_Enemy`, `Safe_Friendly`.

## Results / validation

Generated from the held-out `test` split by:

```bash
.venv/bin/python -m BROWT.detection.report
```

It runs evaluation and rewrites everything between the markers below.

<!-- RESULTS:START -->
_Not generated yet. Run `python -m BROWT.detection.report` after training._
<!-- RESULTS:END -->

Observations: _(weak classes, confusions, sample-count imbalance, ideas to
improve — fill in after reviewing `runs/detect/<name>/`.)_
