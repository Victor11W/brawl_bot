"""Train a YOLOv11 detector on the Brawl Stars dataset.

Usage::

    python -m BROWT.detection.train                 # full run from configs/detection.yaml
    python -m BROWT.detection.train --smoke          # 1-epoch sanity check on a data subset
    python -m BROWT.detection.train --config path.yaml
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from BROWT.detection.dataset import ensure_test_split, write_data_yaml


def repo_root() -> Path:
    """Repository root, derived from this file's location (src layout)."""
    return Path(__file__).resolve().parents[3]


def resolve_device(requested: str) -> str:
    """Map ``auto`` to the best available backend: cuda > mps > cpu."""
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Brawl Stars YOLOv11 detector.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "configs" / "detection.yaml",
        help="Path to the detection training config.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick sanity run: 1 epoch on 5%% of the data to validate the pipeline.",
    )
    args = parser.parse_args()

    # Imported here so --help works without the heavy dependency installed.
    from ultralytics import YOLO

    root = repo_root()
    config = load_config(args.config)

    dataset_root = (root / config["data"]["root"]).resolve()
    counts = ensure_test_split(
        dataset_root,
        fraction=config["data"]["test_fraction"],
        seed=config["data"]["split_seed"],
    )
    data_yaml = write_data_yaml(dataset_root)
    print(f"Dataset splits -> {counts}")

    size = config["model"]["size"]
    weights = config["model"]["pretrained"] or f"yolo11{size}.pt"
    device = resolve_device(config["train"]["device"])
    print(f"Model: {weights} | device: {device}")

    train_cfg = config["train"]
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=1 if args.smoke else train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        patience=train_cfg["patience"],
        device=device,
        workers=train_cfg["workers"],
        seed=train_cfg["seed"],
        project=str(root / config["output"]["project"]),
        name=config["output"]["name"] + ("_smoke" if args.smoke else ""),
        exist_ok=True,
    )
    if args.smoke:
        train_kwargs["fraction"] = 0.05

    model = YOLO(weights)
    model.train(**train_kwargs)

    # Final evaluation on the held-out test split.
    print("Evaluating on the test split...")
    model.val(data=str(data_yaml), split="test", device=device)

    # Publish the best weights for the detector to load.
    run_dir = Path(train_kwargs["project"]) / train_kwargs["name"]
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        weights_dir = root / config["output"]["weights_dir"]
        weights_dir.mkdir(parents=True, exist_ok=True)
        published = weights_dir / f"brawl_yolo11{size}.pt"
        shutil.copy2(best, published)
        print(f"Best weights copied to {published}")


if __name__ == "__main__":
    main()
