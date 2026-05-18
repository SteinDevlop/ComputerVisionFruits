# data/training/detector/train.py — fine-tuning del detector YOLO

from pathlib import Path

from shared.config import BASE_DIR


def cargar_config_entrenamiento() -> dict:
    """Retorna hiperparámetros del entrenamiento.

    Returns:
        dict con epochs, batch_size, lr, etc.
    """
    return {
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 0.001,
        "img_size": 640,
        "data_yaml": str(BASE_DIR / "data" / "training" / "detector" / "dataset.yaml"),
        "model_base": "yolov8n.pt",  # TODO: cambiar a modelo base deseado
        "output_dir": str(BASE_DIR / "models" / "finetuned"),
        "project_name": "fruit_detector",
    }


def entrenar(cfg: dict) -> None:
    """Ejecuta fine-tuning del YOLO.

    Args:
        cfg: dict con hiperparámetros
    """
    # TODO: implementar entrenamiento
    # from ultralytics import YOLO
    # model = YOLO(cfg["model_base"])
    # model.train(
    #     data=cfg["data_yaml"],
    #     epochs=cfg["epochs"],
    #     batch=cfg["batch_size"],
    #     lr0=cfg["learning_rate"],
    #     imgsz=cfg["img_size"],
    #     project=cfg["output_dir"],
    #     name=cfg["project_name"],
    # )
    pass


def validar(model_path: Path, data_yaml: str) -> dict:
    """Valida el modelo entrenado en el split de validacion.

    Args:
        model_path: ruta al .pt
        data_yaml: ruta al yaml del dataset

    Returns:
        dict con metricas mAP50, mAP50-95, etc.
    """
    # TODO: model.val()
    return {}


if __name__ == "__main__":
    cfg = cargar_config_entrenamiento()
    print(f"[Train Detector] Config: {cfg}")
    entrenar(cfg)
