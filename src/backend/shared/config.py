# shared/config.py — config global. todo usa esto.

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class DetectorConfig:
    MODEL_PATH = BASE_DIR / "models" / "finetuned" / "detector_best.pt"
    CLASSIFIER_MODEL_PATH = BASE_DIR / "models" / "finetuned" / "best.pt"
    
    CONFIDENCE_THRESHOLD: float = 0.30  
    IOU_THRESHOLD: float = 0.45
    VIDEO_SOURCE = "data/prueba.mp4" # 0 = camara, o path a video
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    
    # Configuración de detección robusta
    ADAPTIVE_CONFIDENCE: bool = True  # Activa ajuste automático de threshold
    PREPROCESS_FRAMES: bool = True     # Activa mejora de contraste/brillo
    MULTI_SCALE_DETECTION: bool = False  # Detección en múltiples escalas (más lento)

class ClassifierConfig:
    MODEL_PATH = BASE_DIR / "models" / "finetuned" / "best.pt"
    FRUIT_CLASSES: list[str] = [
        "apple", "avocado", "banana", "grapes", "guava", "kiwi",
        "mango", "orange", "peach", "pineapple", "sugarapple", "watermelon"
    ]
    IMAGE_SIZE: int = 640

class APIConfig:
    CLASSIFIER_HOST: str = os.getenv("CLASSIFIER_HOST", "http://localhost")
    CLASSIFIER_PORT: int = int(os.getenv("CLASSIFIER_PORT", "8000"))
    CLASSIFIER_URL: str = os.getenv(
        "CLASSIFIER_URL",
        f"{CLASSIFIER_HOST}:{CLASSIFIER_PORT}/clasificar",
    )
    TIMEOUT_SECONDS: int = int(os.getenv("TIMEOUT_SECONDS", "5"))

detector_cfg = DetectorConfig()
classifier_cfg = ClassifierConfig()
api_cfg = APIConfig()
