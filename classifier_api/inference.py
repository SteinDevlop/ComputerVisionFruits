# classifier_api/inference.py — carga modelo y predice

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from shared.config import classifier_cfg


class FruitClassifier:
    """Wrapper del modelo clasificador fine-tuned."""

    def __init__(self):
        self.model = None
        self.classes = classifier_cfg.CLASSES
        self.image_size = classifier_cfg.IMAGE_SIZE
        self.is_loaded = False

    def load_model(self, model_path: Path | None = None) -> None:
        """Carga el modelo desde disco.

        Args:
            model_path: ruta al .pt. Si None usa config default.
        """
        path = model_path or classifier_cfg.MODEL_PATH
        # TODO: cargar modelo real
        # Ejemplo: self.model = torch.load(path) o timm.create_model(...)
        self.is_loaded = True
        print(f"[Classifier] Modelo cargado desde {path}")

    def preprocess(self, imagen_b64: str) -> np.ndarray:
        """Decodifica base64 y prepara imagen para inferencia.

        Args:
            imagen_b64: imagen codificada en base64

        Returns:
            array numpy listo para el modelo
        """
        # TODO: normalizar segun el modelo (mean/std del preentrenado)
        img_bytes = base64.b64decode(imagen_b64)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    def predict(self, imagen_b64: str) -> tuple[str, float]:
        """Ejecuta inferencia sobre la imagen.

        Args:
            imagen_b64: imagen en base64

        Returns:
            (clase_predicha, confianza)
        """
        if not self.is_loaded:
            raise RuntimeError("Modelo no cargado. Llamar load_model() primero.")

        arr = self.preprocess(imagen_b64)

        # TODO: inferencia real con el modelo
        # logits = self.model(tensor)
        # probs = softmax(logits)
        # idx = argmax(probs)

        # Placeholder
        fruta = self.classes[0]
        confianza = 0.99

        return fruta, confianza


# Singleton — se reutiliza en toda la app
classifier = FruitClassifier()
