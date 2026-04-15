# shared/config.py — config global. todo usa esto.

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class DetectorConfig:
    MODEL_PATH = BASE_DIR / "models" / "finetuned" / "detector_best.pt"
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.45
    VIDEO_SOURCE: int | str = 0  # 0 = camara, o path a video
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720

class ClassifierConfig:
    MODEL_PATH = BASE_DIR / "models" / "finetuned" / "classifier_best.pt"
    CLASSES: list[str] = ["manzana", "banano", "naranja", "pera", "mango"]  # TODO: actualizar
    IMAGE_SIZE: int = 224

class APIConfig:
    CLASSIFIER_HOST: str = "http://classifier_api"
    CLASSIFIER_PORT: int = 8000
    CLASSIFIER_URL: str = f"{CLASSIFIER_HOST}:{CLASSIFIER_PORT}/clasificar"
    TIMEOUT_SECONDS: int = 5

detector_cfg = DetectorConfig()
classifier_cfg = ClassifierConfig()
api_cfg = APIConfig()
