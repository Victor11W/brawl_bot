"""Evaluate the trained detector on the test split and fill the README results.

Usage::

    python -m BROWT.detection.report

Runs validation on the held-out ``test`` split, then rewrites the block
between ``<!-- RESULTS:START -->`` and ``<!-- RESULTS:END -->`` in the
detection README with the metrics tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from BROWT.detection.detector import Detector
from BROWT.detection.train import load_config, repo_root, resolve_device

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def build_markdown(weights: Path, metrics, names: dict[int, str], epochs: int, size: str) -> str:
    box = metrics.box
    speed = metrics.speed  # ms per image: preprocess / inference / postprocess
    infer_ms = speed.get("inference", float("nan"))

    lines = [
        f"Trained model: `{weights.as_posix()}` _(size {size}, imgsz 640, {epochs} epochs configured)_",
        "",
        "Metrics on the held-out `test` split:",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| mAP@50 | {_fmt(box.map50)} |",
        f"| mAP@50-95 | {_fmt(box.map)} |",
        f"| Precision | {_fmt(box.mp)} |",
        f"| Recall | {_fmt(box.mr)} |",
        f"| Inference speed (MPS) | {infer_ms:.1f} ms/image |",
        "",
        "Per-class mAP@50:",
        "",
        "| Class | mAP@50 |",
        "|-------|--------|",
    ]
    # ap_class_index lists the class indices that were present during eval.
    per_class = {int(c): float(ap) for c, ap in zip(metrics.box.ap_class_index, metrics.box.ap50)}
    for cls_id in sorted(names):
        ap50 = per_class.get(cls_id)
        value = _fmt(ap50) if ap50 is not None else "n/a"
        lines.append(f"| {names[cls_id]} | {value} |")
    return "\n".join(lines)


def patch_readme(readme: Path, block: str) -> None:
    text = readme.read_text()
    if START not in text or END not in text:
        raise ValueError(f"Result markers not found in {readme}")
    before = text.split(START)[0]
    after = text.split(END)[1]
    readme.write_text(f"{before}{START}\n{block}\n{END}{after}")


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Evaluate on test and fill the README.")
    parser.add_argument("--config", type=Path, default=root / "configs" / "detection.yaml")
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    size = config["model"]["size"]
    epochs = config["train"]["epochs"]
    weights = args.weights or (root / config["output"]["weights_dir"] / f"brawl_yolo11{size}.pt")

    data_yaml = (root / config["data"]["root"]).resolve() / "data.yaml"
    names = yaml.safe_load(data_yaml.read_text())["names"]
    names = dict(enumerate(names))

    device = resolve_device(config["train"]["device"])

    # Use the detector's underlying model so eval matches inference settings.
    detector = Detector(weights, device=device)
    metrics = detector._model.val(data=str(data_yaml), split="test", device=device, verbose=False)

    block = build_markdown(weights, metrics, names, epochs, size)
    readme = root / "src" / "BROWT" / "detection" / "README.md"
    patch_readme(readme, block)
    print(f"Results written to {readme}")
    print(block)


if __name__ == "__main__":
    main()
