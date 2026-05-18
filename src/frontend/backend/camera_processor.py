"""
frontend/backend/camera_processor.py
Singleton que mantiene YOLO + Tracker persistentes entre frames.
"""
import base64
import sys
import traceback
from io import BytesIO
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Importaciones lazy para no fallar si no están instaladas
_detector = None
_tracker = None
_cropper = None
_client = None
_loaded = False
_load_error: Optional[str] = None

# Paleta de colores para IDs (R,G,B)
BBOX_COLORS = [
    (0, 255, 128),   # verde
    (255, 107, 53),  # naranja
    (99, 102, 241),  # indigo
    (236, 72, 153),  # rosa
    (234, 179, 8),   # amarillo
    (6, 182, 212),   # cyan
    (239, 68, 68),   # rojo
    (168, 85, 247),  # púrpura
]


def color_for_id(obj_id: int):
    return BBOX_COLORS[obj_id % len(BBOX_COLORS)]


def load_pipeline():
    """Carga el pipeline de detección (solo la primera vez)."""
    global _detector, _tracker, _cropper, _client, _loaded, _load_error
    if _loaded:
        return True, None
    if _load_error:
        return False, _load_error
    try:
        from detector_agent.detector import FruitDetector
        from detector_agent.tracker import FruitTracker
        from detector_agent.cropper import Cropper
        from detector_agent.client import ClassifierClient

        _detector = FruitDetector(
            conf_threshold=0.55,         # Threshold más alto → menos falsos positivos
            adaptive_confidence=False,   # DESACTIVADO: bajaba threshold y detectaba caras
            preprocess_frames=True,
            multi_scale=False,
        )
        _detector.load_model()
        _tracker = FruitTracker()
        _cropper = Cropper()
        _client = ClassifierClient()
        _loaded = True
        return True, None
    except Exception as e:
        _load_error = str(e)
        return False, _load_error


def reset_tracker():
    """Resetea el tracker (nueva sesión de cámara)."""
    global _tracker, _loaded, _load_error
    if _tracker:
        _tracker.limpiar()
    _load_error = None


# Caché de clasificaciones: {id_objeto: {fruta, confianza, precio}}
_object_cache: dict = {}
# IDs que fallaron clasificación → reintentar después de N frames
_failed_ids: dict = {}  # {id_objeto: intentos_fallidos}
MAX_RETRIES = 5


def reset_cache():
    global _object_cache, _failed_ids
    _object_cache = {}
    _failed_ids = {}


def process_frame(imagen_b64: str) -> dict:
    """
    Procesa un frame: detecta con YOLO, clasifica con API.
    Retorna lista de detecciones con bbox y clasificación.
    """
    ok, err = load_pipeline()
    if not ok:
        return {"error": err, "detections": []}

    try:
        import cv2
        import numpy as np
        from PIL import Image

        # Decodificar imagen base64
        img_bytes = base64.b64decode(imagen_b64)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        frame = np.array(img)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Detectar
        detecciones = _detector.detect(frame_bgr)
        objetos = _tracker.update(detecciones)

        alto, ancho = frame_bgr.shape[:2]
        frame_area = alto * ancho

        # Umbral mínimo de confianza del clasificador para aceptar resultado
        MIN_CLASSIFIER_CONF = 0.82  # Alto: mejor perder una fruta que clasificar una cara

        results = []
        for obj in objetos:
            x1, y1, x2, y2 = obj.detection.bbox
            obj_id = obj.id_objeto

            # ── Filtro 1: tamaño máximo del bbox
            # Una fruta/verdura raramente ocupa más del 70% del frame.
            # Un bbox enorme casi siempre es una persona o fondo.
            obj_area = (x2 - x1) * (y2 - y1)
            area_ratio = obj_area / frame_area if frame_area > 0 else 1.0
            if area_ratio > 0.45:  # > 45% del frame → demasiado grande para ser fruta
                # Demasiado grande para ser una fruta — mostrar como "No es fruta"
                r, g, b = color_for_id(obj_id)
                results.append({
                    "id": obj_id,
                    "bbox": [max(0,int(x1)), max(0,int(y1)), min(ancho,int(x2)), min(alto,int(y2))],
                    "fruta": "No es fruta",
                    "confianza": round(obj.detection.confidence, 3),
                    "precio": 0,
                    "clasificado": False,
                    "color": [r, g, b],
                })
                continue

            # ── Filtro 2: la confianza del YOLO ya está en 0.55, este es solo guardia adicional
            if obj.detection.confidence < 0.50:
                continue

            # ── Clasificar si no está en caché
            intentos = _failed_ids.get(obj_id, 0)
            if obj_id not in _object_cache and intentos < MAX_RETRIES:
                try:
                    imagen_crop_b64 = _cropper.procesar(frame_bgr, obj)
                    resultado = _client.clasificar(obj_id, imagen_crop_b64)
                    if (resultado
                            and resultado.fruta not in ["Unknown Label", "Unknown", "Desconocida", None]
                            and resultado.confianza >= MIN_CLASSIFIER_CONF):
                        # ✅ Clasificación válida con confianza suficiente
                        _object_cache[obj_id] = {
                            "fruta": resultado.fruta,
                            "confianza": resultado.confianza,
                            "precio": resultado.precio,
                        }
                        obj.clasificado = True
                        _failed_ids.pop(obj_id, None)
                    else:
                        # Baja confianza o fruta inválida → reintentar hasta MAX_RETRIES
                        _failed_ids[obj_id] = intentos + 1
                except Exception:
                    _failed_ids[obj_id] = intentos + 1

            info = _object_cache.get(obj_id, {})
            r, g, b = color_for_id(obj_id)
            results.append({
                "id": obj_id,
                "bbox": [
                    max(0, int(x1)), max(0, int(y1)),
                    min(ancho, int(x2)), min(alto, int(y2))
                ],
                "fruta": info.get("fruta", "Detectando..."),
                "confianza": round(info.get("confianza", obj.detection.confidence), 3),
                "precio": info.get("precio", 0),
                "clasificado": obj.clasificado,
                "color": [r, g, b],
            })

        return {"detections": results, "error": None}

    except Exception as e:
        return {"error": str(e), "detections": [], "traceback": traceback.format_exc()}
