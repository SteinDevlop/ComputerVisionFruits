# shared/config.py — config global. todo usa esto.

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class DetectorConfig:
    MODEL_PATH = BASE_DIR / "models" / "finetuned" / "detector_best.pt"
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.45
    VIDEO_SOURCE = "data/prueba.mp4" # 0 = camara, o path a video
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720

class ClassifierConfig:
    MODEL_PATH = BASE_DIR / "models" / "finetuned" / "classifier_best.pt"
    HF_MODEL_ID: str = "bhumong/fruit-classifier-efficientnet-b0"
    CLASSES: list[str] = [
        "Apple", "Apricot", "Avocado", "Banana", "Beans", "Beetroot", "Blackberry",
        "Blueberry", "Cabbage", "Cactus", "Caju", "Cantaloupe", "Carambula", "Carrot",
        "Cauliflower", "Cherry", "Chestnut", "Clementine", "Cocona", "Corn", "Cucumber",
        "Dates", "Eggplant", "Fig", "Ginger", "Gooseberry", "Granadilla", "Grape",
        "Grapefruit", "Guava", "Hazelnut", "Huckleberry", "Kiwi", "Kohlrabi", "Lemon",
        "Limes", "Lychee", "Mango", "Mangostan", "Maracuja", "Melon", "Mulberry",
        "Nectarine", "Nut", "Onion", "Orange", "Papaya", "Passion", "Peach", "Pear",
        "Pepino", "Pepper", "Physalis", "Pineapple", "Pistachio", "Pitahaya", "Plum",
        "Pomegranate", "Potato", "Quince", "Rambutan", "Raspberry", "Redcurrant",
        "Salak", "Strawberry", "Tamarillo", "Tangelo", "Tomato", "Walnut", "Watermelon"
    ]
    IMAGE_SIZE: int = 224

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
