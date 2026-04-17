# detector_agent/detector.py

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from ultralytics import YOLO

from shared.config import detector_cfg


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int


class FruitDetector:

    def __init__(self):
        self.model: YOLO | None = None
        self.conf_threshold = detector_cfg.CONFIDENCE_THRESHOLD
        self.iou_threshold = detector_cfg.IOU_THRESHOLD

    def load_model(self, model_path: Path | None = None) -> None:
        """
        Carga YOLO de forma compatible con PyTorch 2.1
        """

        path = model_path or detector_cfg.MODEL_PATH

        print("[Detector] Cargando YOLO...")


        self.model = YOLO(str(path))

        print(f"[Detector] Modelo cargado desde {path}")

    def detect(self, frame: np.ndarray) -> List[Detection]:

        if self.model is None:
            raise RuntimeError("Modelo no cargado.")

        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        detections: List[Detection] = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls
                    )
                )

        return detections