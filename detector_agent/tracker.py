# detector_agent/tracker.py — asigna ID persistente a cada objeto

from dataclasses import dataclass, field

import numpy as np

from detector_agent.detector import Detection


@dataclass
class TrackedObject:
    """Objeto con ID de tracking asignado."""
    id_objeto: int
    detection: Detection
    frames_sin_ver: int = 0  # cuantos frames no fue detectado


class FruitTracker:
    """Tracker simple basado en IoU. Reemplazar con ByteTrack/DeepSORT si se necesita."""

    def __init__(self, iou_threshold: float = 0.3, max_frames_perdido: int = 5):
        self.iou_threshold = iou_threshold
        self.max_frames_perdido = max_frames_perdido
        self._next_id: int = 0
        self._objetos: dict[int, TrackedObject] = {}

    def _siguiente_id(self) -> int:
        """Genera ID único incremental."""
        id_ = self._next_id
        self._next_id += 1
        return id_

    def _calcular_iou(
        self, bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]
    ) -> float:
        """Calcula Intersection over Union entre dos bboxes.

        Args:
            bbox_a: (x1, y1, x2, y2)
            bbox_b: (x1, y1, x2, y2)

        Returns:
            IoU entre 0.0 y 1.0
        """
        # TODO: implementar IoU
        pass

    def update(self, detecciones: list[Detection]) -> list[TrackedObject]:
        """Asocia detecciones actuales con objetos trackeados.

        Args:
            detecciones: lista de Detection del frame actual

        Returns:
            lista de TrackedObject con IDs asignados
        """
        # TODO: matching por IoU entre detecciones y objetos activos
        # 1. Calcular matriz IoU
        # 2. Asignar detecciones a objetos existentes
        # 3. Crear nuevos objetos para detecciones sin match
        # 4. Incrementar frames_sin_ver para objetos no vistos
        # 5. Eliminar objetos perdidos por mas de max_frames_perdido

        resultado: list[TrackedObject] = []
        return resultado

    def limpiar(self) -> None:
        """Reinicia el estado del tracker."""
        self._objetos.clear()
        self._next_id = 0
