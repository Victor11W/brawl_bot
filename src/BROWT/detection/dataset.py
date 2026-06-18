"""Dataset preparation for YOLOv11 training.

The Roboflow export ships only ``train`` and ``valid`` splits and a
``data.yaml`` whose relative paths do not resolve from the file location.
This module:

1. Carves a deterministic ``test`` split out of ``train`` when none exists.
2. Writes a clean ``data.yaml`` with an absolute ``path`` root so Ultralytics
   resolves every split unambiguously.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _image_label_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Return (image, label) pairs sorted by filename for determinism."""
    pairs: list[tuple[Path, Path]] = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = labels_dir / f"{image.stem}.txt"
        if label.exists():
            pairs.append((image, label))
    return pairs


def ensure_test_split(root: Path, fraction: float, seed: int) -> dict[str, int]:
    """Create a ``test`` split from ``train`` if it does not already exist.

    Selection is deterministic: pairs are sorted by filename and every
    ``round(1 / fraction)``-th pair (offset by ``seed``) is moved to ``test``.
    The function is idempotent -- if ``test`` already holds images, it is a
    no-op.

    Returns the image count of each split.
    """
    train_images = root / "train" / "images"
    train_labels = root / "train" / "labels"
    test_images = root / "test" / "images"
    test_labels = root / "test" / "labels"

    existing_test = (
        [p for p in test_images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        if test_images.is_dir()
        else []
    )

    if not existing_test:
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"test_fraction must be in (0, 1), got {fraction}")

        test_images.mkdir(parents=True, exist_ok=True)
        test_labels.mkdir(parents=True, exist_ok=True)

        pairs = _image_label_pairs(train_images, train_labels)
        stride = max(2, round(1 / fraction))
        offset = seed % stride
        selected = pairs[offset::stride]

        for image, label in selected:
            shutil.move(str(image), str(test_images / image.name))
            shutil.move(str(label), str(test_labels / label.name))

    return {
        "train": _count_images(train_images),
        "valid": _count_images(root / "valid" / "images"),
        "test": _count_images(test_images),
    }


def _count_images(images_dir: Path) -> int:
    if not images_dir.is_dir():
        return 0
    return sum(1 for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def write_data_yaml(root: Path) -> Path:
    """Write a clean ``data.yaml`` with an absolute root path.

    Class names are read from the original Roboflow ``data.yaml`` so the class
    index order matches the label files. The resolved file is written back to
    ``<root>/data.yaml`` and its path returned.
    """
    source = root / "data.yaml"
    if not source.exists():
        raise FileNotFoundError(f"Missing Roboflow data.yaml at {source}")

    original = yaml.safe_load(source.read_text())
    names = original["names"]

    config = {
        "path": str(root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    source.write_text(yaml.safe_dump(config, sort_keys=False))
    return source
