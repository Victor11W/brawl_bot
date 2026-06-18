"""Inference wrapper around a trained YOLOv11 detector.

Gives the rest of the pipeline a single entry point: feed an image, get a
list of :class:`Detection` (class, confidence, bounding box). Decoupled from
how the weights were produced so the RL/state layers never touch Ultralytics
directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

ImageInput = Union[str, Path, np.ndarray]


@dataclass(frozen=True)
class Detection:
    """A single detected object."""

    cls_id: int
    cls_name: str
    confidence: float
    # Absolute pixel box, (x1, y1, x2, y2).
    xyxy: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class Detector:
    """Loads a trained model and runs detection on frames."""

    def __init__(
        self,
        weights: Union[str, Path],
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "auto",
        imgsz: int = 640,
    ) -> None:
        from ultralytics import YOLO

        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"Weights not found at {weights}. Train first with "
                "`python -m BROWT.detection.train`."
            )

        self._model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = self._resolve_device(device)
        self.class_names: dict[int, str] = self._model.names

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested != "auto":
            return requested
        import torch

        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def detect(self, image: ImageInput) -> list[Detection]:
        """Run detection on a single image and return structured results."""
        results = self._model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        detections: list[Detection] = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(
                Detection(
                    cls_id=cls_id,
                    cls_name=self.class_names[cls_id],
                    confidence=float(box.conf[0]),
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return detections

    def __call__(self, image: ImageInput) -> list[Detection]:
        return self.detect(image)
