# detector_agent/detector.py
# Módulo detector: Detección robusta de frutas usando YOLO

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import warnings

import numpy as np
import cv2
import torch
from ultralytics import YOLO

from shared.config import detector_cfg
from detector_agent.logger_config import setup_logger

# Suprimir warnings de YOLO
warnings.filterwarnings("ignore", category=UserWarning)

logger = setup_logger(__name__)


@dataclass
class Detection:
    """
    Representa una detección de fruta en el video.
    
    Attributes:
        bbox (tuple): Bounding box (x1, y1, x2, y2) en píxeles
        confidence (float): Confianza de la detección (0.0 a 1.0)
        class_id (int): ID de la clase del objeto detectado
    """
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int


class FruitDetector:
    """
    Detector robusto de frutas usando YOLOv8n con preprocesamiento adaptativo.
    
    Soporta:
    - Ajuste dinámico de umbral de confianza
    - Preprocesamiento de imágenes (contraste, brillo)
    - Detección multi-escala opcional
    
    Attributes:
        model: Modelo YOLO cargado
        conf_threshold: Umbral inicial de confianza
        iou_threshold: Umbral de IoU para NMS
        adaptive_confidence: Si activa ajuste dinámico
        preprocess_frames: Si activa preprocesamiento
        multi_scale: Si activa detección multi-escala
    """

    def __init__(
        self,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        adaptive_confidence: bool = True,
        preprocess_frames: bool = True,
        multi_scale: bool = False
    ):
        """
        Inicializa el detector.
        
        Args:
            conf_threshold: Umbral de confianza inicial (0.0-1.0)
            iou_threshold: Umbral de IoU para supresión de duplicados (0.0-1.0)
            adaptive_confidence: Habilitar ajuste automático del umbral
            preprocess_frames: Habilitar preprocesamiento de imágenes
            multi_scale: Habilitar detección en múltiples escalas
        """
        self.model: YOLO | None = None
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.adaptive_confidence = adaptive_confidence
        self.preprocess_frames = preprocess_frames
        self.multi_scale = multi_scale

        # Para detección adaptativa
        self.frames_without_detection = 0
        self.current_conf_threshold = conf_threshold
        self.min_conf_threshold = 0.25  # No bajar más de esto
        self.detection_streak = 0

    def load_model(self, model_path: Path | None = None) -> None:
        """
        Carga el modelo YOLO desde disco de forma compatible con PyTorch 2.1.
        
        Realiza validación del modelo después de cargarlo ejecutando
        una detección prueba en un frame dummy.
        
        Args:
            model_path: Ruta al archivo del modelo. Si es None, usa config.MODEL_PATH
            
        Raises:
            RuntimeError: Si no se puede cargar o validar el modelo
        """
        path = model_path or detector_cfg.MODEL_PATH

        logger.info(f"Cargando modelo YOLO desde {path}...")

        try:
            self.model = YOLO(str(path))
            logger.info("Modelo cargado correctamente")

            # Validar que el modelo se cargó correctamente
            if self.model is None:
                raise RuntimeError("Modelo YOLO es None después de cargar")

            logger.debug("Ejecutando prueba de detección en frame dummy...")
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy, conf=0.5, verbose=False)
            logger.info("Modelo validado exitosamente")

        except Exception as e:
            logger.error(f"Error cargando modelo: {e}", exc_info=True)
            raise

    def _adjust_confidence_threshold(self, detections_found: int) -> None:
        """
        Ajusta dinámicamente el umbral de confianza si no hay detecciones.
        
        Estrategia adaptativa:
        - Si no hay detecciones en 3 frames consecutivos, baja el umbral 0.05
        - Si hay muchas detecciones seguidas, sube el umbral lentamente
        - No baja del umbral mínimo (0.25)
        
        Args:
            detections_found: Número de detecciones en el frame actual
        """
        if not self.adaptive_confidence:
            return

        if detections_found == 0:
            self.frames_without_detection += 1

            # Cada 3 frames sin detecciones, bajar el umbral
            if self.frames_without_detection >= 3:
                new_threshold = max(
                    self.min_conf_threshold,
                    self.current_conf_threshold - 0.05
                )
                if new_threshold != self.current_conf_threshold:
                    logger.info(
                        f"Sin detecciones por 3 frames. "
                        f"Ajustando threshold: {self.current_conf_threshold:.3f} → {new_threshold:.3f}"
                    )
                    self.current_conf_threshold = new_threshold
                    self.frames_without_detection = 0

        else:
            # Se encontraron detecciones
            self.detection_streak += 1

            # Si hay varias detecciones seguidas, podemos aumentar el umbral
            if self.detection_streak > 10 and self.current_conf_threshold < self.conf_threshold:
                self.current_conf_threshold = min(
                    self.conf_threshold,
                    self.current_conf_threshold + 0.02
                )
                self.detection_streak = 0

            # Reset counter
            self.frames_without_detection = 0

    def _detect_single_scale(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None
    ) -> List[Detection]:
        """
        Realiza detección YOLO en una única escala.
        
        Args:
            frame: Frame de entrada (imagen numpy)
            conf_threshold: Umbral de confianza. Si es None, usa el actual
            
        Returns:
            Lista de detecciones encontradas
            
        Raises:
            RuntimeError: Si el modelo no está cargado
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado. Llama a load_model() primero")

        threshold = conf_threshold or self.current_conf_threshold

        try:
            results = self.model(
                frame,
                conf=threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            detections: List[Detection] = []

            for r in results:
                if r.boxes is None:
                    continue

                for box in r.boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])

                        # Validar coordenadas
                        if x2 > x1 and y2 > y1:
                            detections.append(
                                Detection(
                                    bbox=(x1, y1, x2, y2),
                                    confidence=conf,
                                    class_id=cls
                                )
                            )
                    except Exception as e:
                        logger.debug(f"Error procesando box: {e}")
                        continue

            return detections

        except Exception as e:
            logger.error(f"Error en detección: {e}", exc_info=True)
            return []

    def _detect_multi_scale(self, frame: np.ndarray) -> List[Detection]:
        """
        Realiza detección en múltiples escalas (escala original y mitad).
        
        Útil para detectar objetos pequeños. Ejecuta detección en el frame
        original y en una versión reducida a la mitad, escalando coordenadas
        de vuelta a la resolución original.
        
        Args:
            frame: Frame de entrada
            
        Returns:
            Lista combinada de detecciones en ambas escalas
        """
        detections = self._detect_single_scale(frame, self.current_conf_threshold)

        # Intentar detección en escala reducida (útil para objetos pequeños)
        try:
            h, w = frame.shape[:2]
            if h > 640 or w > 640:
                # Reducir a la mitad solo si frame es grande
                frame_half = cv2.resize(frame, (w // 2, h // 2))
                detections_half = self._detect_single_scale(frame_half)

                # Escalar coordenadas de vuelta
                for det in detections_half:
                    x1, y1, x2, y2 = det.bbox
                    scaled_bbox = (x1 * 2, y1 * 2, x2 * 2, y2 * 2)
                    detections.append(
                        Detection(
                            bbox=scaled_bbox,
                            confidence=det.confidence,
                            class_id=det.class_id
                        )
                    )

        except Exception as e:
            logger.debug(f"Error en multi-scale: {e}")

        return detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Realiza detección robusta con preprocesamiento y estrategia adaptativa.
        
        Proceso:
        1. Valida que el frame sea válido
        2. Preprocesa el frame si está habilitado (contraste, brillo)
        3. Ejecuta detección (escala única o múltiple)
        4. Ajusta umbral adaptativo si es necesario
        
        Args:
            frame: Frame de entrada (imagen numpy BGR)
            
        Returns:
            Lista de detecciones encontradas. Lista vacía si hay error.
        """
        if frame is None or frame.size == 0:
            logger.warning("Frame vacío o None recibido")
            return []

        try:
            # Ejecutar detección
            if self.multi_scale:
                detections = self._detect_multi_scale(frame)
            else:
                detections = self._detect_single_scale(frame)

            # Ajustar umbral adaptativo
            self._adjust_confidence_threshold(len(detections))

            return detections

        except Exception as e:
            logger.error(f"Error en detect: {e}", exc_info=True)
            return []

    def get_current_threshold(self) -> float:
        """
        Retorna el umbral de confianza actual.
        
        Este valor puede cambiar dinámicamente si está habilitada
        la estrategia adaptativa.
        
        Returns:
            float: Umbral de confianza actual (0.0 a 1.0)
        """
        return self.current_conf_threshold