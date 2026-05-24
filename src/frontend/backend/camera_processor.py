"""
frontend/backend/camera_processor.py
Pipeline de 1 etapa para detección y clasificación de frutas:
  1. best.pt (YOLOv8 unificado) -> bounding boxes + clases
"""
import base64
import sys
import traceback
from io import BytesIO
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_detector = None
_tracker = None
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
    """Carga el modelo YOLO unificado (solo la primera vez)."""
    global _detector, _tracker, _loaded, _load_error
    if _loaded:
        return True, None
    if _load_error:
        return False, _load_error
    try:
        from detector_agent.detector import FruitDetector
        from detector_agent.tracker import FruitTracker
        from shared.config import detector_cfg

        # Etapa única: Detector YOLOv8 unificado (bboxes + clase)
        _detector = FruitDetector(
            conf_threshold=0.30,
            adaptive_confidence=True,
            preprocess_frames=True,
            multi_scale=False,
        )
        
        # Cargar el modelo best.pt unificado
        unified_model_path = PROJECT_ROOT / "backend" / "models" / "finetuned" / "best.pt"
        if not unified_model_path.exists():
            unified_model_path = PROJECT_ROOT / "models" / "finetuned" / "best.pt"
            
        _detector.load_model(model_path=unified_model_path)

        _tracker = FruitTracker()
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


def process_frame(imagen_b64: str) -> dict:
    """
    Procesa un frame con pipeline unificado:
      1. Detectar objetos y clase con best.pt
    """
    ok, err = load_pipeline()
    if not ok:
        return {"error": err, "detections": []}

    try:
        import cv2
        import json
        import numpy as np
        from PIL import Image

        # Cargar base de datos de precios
        precios_db = {}
        for candidate in [
            PROJECT_ROOT / "backend" / "database" / "precios.json",
            PROJECT_ROOT / "database" / "precios.json",
        ]:
            if candidate.exists():
                with open(candidate, "r", encoding="utf-8") as f:
                    precios_db = json.load(f)
                break

        # Decodificar imagen base64
        img_bytes = base64.b64decode(imagen_b64)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        frame = np.array(img)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # ── Etapa Única: Detectar y clasificar con YOLO unificado ──
        detecciones = _detector.detect(frame_bgr)
        objetos = _tracker.update(detecciones)

        alto, ancho = frame_bgr.shape[:2]
        results = []

        for obj in objetos:
            x1, y1, x2, y2 = obj.detection.bbox
            obj_id = obj.id_objeto
            conf = obj.detection.confidence

            if conf < 0.30:
                continue
                
            # Extraer la clase directamente del modelo YOLO
            class_id = obj.detection.class_id
            if hasattr(_detector.model, 'names'):
                yolo_label = _detector.model.names[class_id]
            else:
                yolo_label = f"Clase_{class_id}"
                
            # Mapear la etiqueta al nombre capitalizado para precios.json
            fruta = yolo_label.capitalize()
            precio = precios_db.get(fruta, 0)
            
            obj.clasificado = True
            obj.etiqueta = fruta
            obj.confianza = conf
            obj.precio = precio

            r, g, b = color_for_id(obj_id)
            results.append({
                "id": obj_id,
                "bbox": [
                    max(0, int(x1)), max(0, int(y1)),
                    min(ancho, int(x2)), min(alto, int(y2))
                ],
                "fruta": fruta,
                "confianza": round(conf, 3),
                "precio": precio,
                "clasificado": True,
                "color": [r, g, b],
            })

        return {"detections": results, "error": None}

    except Exception as e:
        return {"error": str(e), "detections": [], "traceback": traceback.format_exc()}
