# detector_agent/detector.py — carga YOLO y detecta objetos

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared.config import detector_cfg


@dataclass
class Detection:
    """Un objeto detectado en un frame."""
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int


class FruitDetector:
    """Wrapper YOLO fine-tuned para detección de frutas."""

    def __init__(self):
        self.model = None
        self.conf_threshold = detector_cfg.CONFIDENCE_THRESHOLD
        self.iou_threshold = detector_cfg.IOU_THRESHOLD

    def load_model(self, model_path: Path | None = None) -> None:
        """Carga el modelo YOLO desde disco.

        Args:
            model_path: ruta al .pt. Si None usa config default.
        """
        path = model_path or detector_cfg.MODEL_PATH
        # TODO: cargar modelo YOLO
        # from ultralytics import YOLO
        # self.model = YOLO(str(path))
        print(f"[Detector] Modelo cargado desde {path}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Corre inferencia YOLO en un frame.

        Args:
            frame: imagen BGR de OpenCV (H, W, 3)

        Returns:
            lista de Detection con bbox y confianza
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado.")

        detections: list[Detection] = []

        # TODO: inferencia real
        # results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
        # for box in results[0].boxes:
        #     bbox = tuple(map(int, box.xyxy[0]))
        #     detections.append(Detection(bbox, float(box.conf), int(box.cls)))

        return detections
