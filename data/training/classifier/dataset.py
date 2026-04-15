# data/training/classifier/dataset.py — dataset para clasificacion de frutas

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from shared.config import classifier_cfg


class FruitDataset(Dataset):
    """Dataset PyTorch para clasificacion de frutas.

    Estructura de carpetas esperada:
        root/
            train/
                manzana/img1.jpg ...
                banano/img1.jpg ...
            val/
                manzana/ ...
    """

    def __init__(self, root: Path, split: str = "train", transform=None):
        """
        Args:
            root: directorio raiz del dataset
            split: "train" o "val"
            transform: transformaciones torchvision (opcional)
        """
        self.root = root / split
        self.transform = transform
        self.clases = classifier_cfg.CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.clases)}
        self.samples: list[tuple[Path, int]] = []

        self._cargar_muestras()

    def _cargar_muestras(self) -> None:
        """Escanea carpetas y llena self.samples con (path, label)."""
        # TODO: recorrer self.root, leer imagen por clase
        for clase in self.clases:
            clase_dir = self.root / clase
            if not clase_dir.exists():
                continue
            for img_path in clase_dir.glob("*.jpg"):
                self.samples.append((img_path, self.class_to_idx[clase]))
            # TODO: agregar .png, .jpeg, .webp si aplica

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        """Retorna (imagen_tensor, label).

        Args:
            idx: indice de muestra

        Returns:
            (tensor imagen, int label)
        """
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


def obtener_transformaciones(split: str, image_size: int = 224):
    """Retorna transformaciones torchvision para train o val.

    Args:
        split: "train" o "val"
        image_size: tamaño de imagen de entrada al modelo

    Returns:
        transforms.Compose
    """
    # TODO: implementar
    # from torchvision import transforms
    # if split == "train":
    #     return transforms.Compose([
    #         transforms.RandomResizedCrop(image_size),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ColorJitter(...),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    # else: solo Resize + CenterCrop + ToTensor + Normalize
    pass
