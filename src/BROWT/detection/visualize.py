"""Draw detector predictions on images for a visual quality check.

Usage::

    # Sample random images from the test split
    python -m BROWT.detection.visualize

    # A specific image or a directory
    python -m BROWT.detection.visualize --source path/to/frame.png
    python -m BROWT.detection.visualize --source data/dataset/test/images --num 12

Annotated images are written to ``runs/viz/`` by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from BROWT.detection.detector import Detection, Detector

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Distinct BGR colors, one per class index (10 classes).
_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (199, 55, 255), (255, 112, 31),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _color(cls_id: int) -> tuple[int, int, int]:
    return _PALETTE[cls_id % len(_PALETTE)]


def draw(image, detections: list[Detection]):
    """Draw boxes and labels onto a BGR image (in place) and return it."""
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        color = _color(det.cls_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"{det.cls_name} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - baseline - 2), (x1 + tw, y1), color, -1)
        cv2.putText(
            image, label, (x1, y1 - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return image


def _collect_sources(source: Path | None, num: int) -> list[Path]:
    root = repo_root()
    if source is None:
        test_images = root / "data" / "dataset" / "test" / "images"
        images = sorted(p for p in test_images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        # Deterministic spread across the split rather than the first N.
        stride = max(1, len(images) // num)
        return images[::stride][:num]
    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        return images[:num]
    return [source]


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Visualize detector predictions.")
    parser.add_argument("--weights", type=Path, default=root / "data" / "models" / "brawl_yolo11n.pt")
    parser.add_argument("--source", type=Path, default=None, help="Image or directory. Default: test split.")
    parser.add_argument("--num", type=int, default=8, help="Number of images when sampling.")
    parser.add_argument("--output", type=Path, default=root / "runs" / "viz")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    detector = Detector(args.weights, conf=args.conf)
    sources = _collect_sources(args.source, args.num)
    args.output.mkdir(parents=True, exist_ok=True)

    for path in sources:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipped unreadable image: {path}")
            continue
        detections = detector.detect(path)
        draw(image, detections)
        out_path = args.output / f"{path.stem}_pred.jpg"
        cv2.imwrite(str(out_path), image)
        print(f"{path.name}: {len(detections)} detections -> {out_path}")


if __name__ == "__main__":
    main()
