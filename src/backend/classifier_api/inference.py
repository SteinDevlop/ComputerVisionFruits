import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torchvision.models as models
from torchvision import transforms
import json
from huggingface_hub import hf_hub_download

from shared.config import classifier_cfg


class FruitClassifier:
    """Wrapper del modelo clasificador fine-tuned."""

    def __init__(self):
        self.model = None
        self.classes = classifier_cfg.CLASSES
        self.image_size = classifier_cfg.IMAGE_SIZE
        self.is_loaded = False
        self.id2label = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.preprocess_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def load_model(self, model_path: Path | None = None) -> None:
        """Carga el modelo desde disco o Hugging Face Hub.

        Args:
            model_path: ruta al .pt. Si None usa config default.
        """
        if model_path:
            # Cargar desde path local si se proporciona
            path = model_path
            raise NotImplementedError("Carga de modelo local no implementada para este ejemplo.")
        else:
            # Cargar desde Hugging Face Hub
            repo_id = classifier_cfg.HF_MODEL_ID
            print(f"[Classifier] Descargando modelo y config de Hugging Face Hub: {repo_id}")

            # Descargar config file
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)

            num_labels = config['num_labels']
            self.id2label = {int(k): v for k, v in config['id2label'].items()} # Convert keys to int

            # Instanciar la arquitectura correcta (EfficientNet-B0)
            model = models.efficientnet_b0(weights=None) # Cargar arquitectura sin pesos pre-entrenados

            # Modificar la cabeza del clasificador para que coincida con el número de clases
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(num_ftrs, num_labels)

            # Descargar pesos del modelo
            model_weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")

            # Cargar el estado del diccionario
            state_dict = torch.load(model_weights_path, map_location=self.device)
            model.load_state_dict(state_dict)

            model.eval() # Establecer en modo de evaluación
            self.model = model.to(self.device)

            print(f"[Classifier] Modelo cargado exitosamente desde {repo_id} y listo para inferencia.")

        self.is_loaded = True

    def preprocess(self, imagen_b64: str) -> torch.Tensor:
        """Decodifica base64 y prepara imagen para inferencia.

        Args:
            imagen_b64: imagen codificada en base64

        Returns:
            tensor de PyTorch listo para el modelo
        """
        img_bytes = base64.b64decode(imagen_b64)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        input_tensor = self.preprocess_transform(img)
        # Añadir dimensión de batch (el modelo espera batches)
        return input_tensor.unsqueeze(0).to(self.device)

    def predict(self, imagen_b64: str) -> tuple[str, float]:
        """Ejecuta inferencia sobre la imagen.

        Args:
            imagen_b64: imagen en base64

        Returns:
            (clase_predicha, confianza)
        """
        if not self.is_loaded:
            raise RuntimeError("Modelo no cargado. Llamar load_model() primero.")

        input_batch = self.preprocess(imagen_b64)

        with torch.no_grad(): # Deshabilitar cálculos de gradiente para inferencia
            output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_catid = torch.max(probabilities, dim=0)

        predicted_label_index = top_catid.item()
        predicted_label = self.id2label.get(predicted_label_index, "Unknown Label")
        confidence = top_prob.item()

        return predicted_label, confidence


classifier = FruitClassifier()
