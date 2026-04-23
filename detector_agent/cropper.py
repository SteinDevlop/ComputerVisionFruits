"""
Módulo cropper: Recorte y preprocesamiento de regiones de interés.

Extrae regiones de imágenes correspondientes a detecciones
y las convierte a formato base64 para envío por HTTP.
"""

import base64
import cv2
import numpy as np
from detector_agent.tracker import TrackedObject
from detector_agent.logger_config import setup_logger

logger = setup_logger(__name__)


class Cropper:
    """
    Recorta regiones de interés de frames y las convierte a base64.
    
    Attributes:
        padding: Píxeles de margen a añadir alrededor del bounding box
    """

    def __init__(self, padding=10):
        """
        Inicializa el cropper.
        
        Args:
            padding: Píxeles de margen alrededor del bounding box
        """
        self.padding = padding
        logger.debug(f"Cropper inicializado con padding={padding}")

    def recortar(self, frame, obj):
        """
        Recorta una región de interés del frame.
        
        Args:
            frame: Frame completo (numpy array)
            obj: Objeto TrackedObject con bbox
            
        Returns:
            np.ndarray: Región recortada del frame
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = obj.detection.bbox

        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)
        x2 = min(w, x2 + self.padding)
        y2 = min(h, y2 + self.padding)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            logger.warning(f"Crop vacío para objeto {obj.id_objeto}")
            return frame

        return crop

    def a_base64(self, img):
        """
        Codifica una imagen a base64 (JPEG 85% calidad).
        
        Args:
            img: Imagen numpy array
            
        Returns:
            str: String base64 de la imagen
            
        Raises:
            RuntimeError: Si no se puede codificar la imagen
        """
        try:
            ok, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])

            if not ok:
                raise RuntimeError("Error encoding image to JPEG")

            return base64.b64encode(buffer).decode()
        except Exception as e:
            logger.error(f"Error codificando imagen a base64: {e}")
            raise

    def procesar(self, frame, obj):
        """
        Realiza el pipeline completo: recortar y codificar a base64.
        
        Args:
            frame: Frame completo
            obj: Objeto TrackedObject
            
        Returns:
            str: String base64 de la región recortada
        """
        try:
            crop = self.recortar(frame, obj)
            return self.a_base64(crop)
        except Exception as e:
            logger.error(f"Error procesando crop para objeto {obj.id_objeto}: {e}")
            raise