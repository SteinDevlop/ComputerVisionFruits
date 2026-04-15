# detector_agent/cropper.py — recorta ROI del frame segun bbox

import base64

import cv2
import numpy as np

from detector_agent.tracker import TrackedObject


class Cropper:
    """Recorta y codifica la región de interés de cada objeto."""

    def __init__(self, padding: int = 10):
        """
        Args:
            padding: pixeles extra alrededor del bbox (evita cortes)
        """
        self.padding = padding

    def recortar(self, frame: np.ndarray, obj: TrackedObject) -> np.ndarray:
        """Recorta el bbox del objeto en el frame con padding.

        Args:
            frame: imagen BGR completa (H, W, 3)
            obj: objeto trackeado con bbox

        Returns:
            recorte BGR del objeto
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = obj.detection.bbox
        p = self.padding

        # Clamping para no salirse del frame
        x1 = max(0, x1 - p)
        y1 = max(0, y1 - p)
        x2 = min(w, x2 + p)
        y2 = min(h, y2 + p)

        return frame[y1:y2, x1:x2]

    def a_base64(self, recorte: np.ndarray) -> str:
        """Convierte recorte BGR a string base64 JPEG.

        Args:
            recorte: imagen BGR numpy

        Returns:
            string base64 listo para enviar al clasificador
        """
        # TODO: ajustar calidad JPEG si se necesita menor payload
        _, buffer = cv2.imencode(".jpg", recorte, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode("utf-8")

    def procesar(self, frame: np.ndarray, obj: TrackedObject) -> str:
        """Recorta y codifica en un paso.

        Args:
            frame: frame completo
            obj: objeto trackeado

        Returns:
            imagen base64
        """
        recorte = self.recortar(frame, obj)
        return self.a_base64(recorte)
