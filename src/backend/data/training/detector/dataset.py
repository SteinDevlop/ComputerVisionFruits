# data/training/detector/dataset.py — dataset YOLO para deteccion

from pathlib import Path

import yaml


# dataset.yaml esperado por YOLO — generar o verificar este archivo antes de entrenar
DATASET_YAML_TEMPLATE = {
    "path": "/data/fruit_detection",   # TODO: ruta real
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 1,                           # TODO: numero de clases
    "names": ["fruta"],                # TODO: lista real de clases
}


def generar_yaml(output_path: Path) -> None:
    """Genera el dataset.yaml necesario para entrenar YOLO.

    Args:
        output_path: donde guardar el yaml
    """
    # TODO: ajustar rutas y clases antes de usar
    with open(output_path, "w") as f:
        yaml.dump(DATASET_YAML_TEMPLATE, f, default_flow_style=False)
    print(f"[Dataset] YAML generado en {output_path}")


def verificar_estructura(data_dir: Path) -> bool:
    """Verifica que el directorio tenga la estructura correcta para YOLO.

    Estructura esperada:
        data_dir/
            images/train/
            images/val/
            labels/train/
            labels/val/

    Args:
        data_dir: directorio raiz del dataset

    Returns:
        True si estructura es válida
    """
    # TODO: verificar existencia de carpetas y que haya coincidencia imagen/label
    required = [
        data_dir / "images" / "train",
        data_dir / "images" / "val",
        data_dir / "labels" / "train",
        data_dir / "labels" / "val",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print(f"[Dataset] Faltan carpetas: {missing}")
        return False
    return True


def contar_muestras(data_dir: Path) -> dict:
    """Cuenta imagenes por split.

    Args:
        data_dir: directorio raiz

    Returns:
        dict { "train": N, "val": N, "test": N }
    """
    # TODO: contar archivos reales
    return {"train": 0, "val": 0, "test": 0}
