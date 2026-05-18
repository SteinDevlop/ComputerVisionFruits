# data/training/classifier/train.py — fine-tuning del clasificador de frutas

from pathlib import Path

from shared.config import BASE_DIR, classifier_cfg


def cargar_config_entrenamiento() -> dict:
    """Retorna hiperparámetros del clasificador.

    Returns:
        dict con epochs, batch_size, lr, arquitectura, etc.
    """
    return {
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "image_size": classifier_cfg.IMAGE_SIZE,
        "num_classes": len(classifier_cfg.CLASSES),
        "classes": classifier_cfg.CLASSES,
        "base_model": "efficientnet_b0",   # TODO: cambiar si se usa otra arquitectura
        "data_dir": str(BASE_DIR / "data" / "training" / "classifier" / "images"),
        "output_dir": str(BASE_DIR / "models" / "finetuned"),
        "model_name": "classifier_best.pt",
    }


def construir_modelo(cfg: dict):
    """Crea arquitectura base con cabeza de clasificacion custom.

    Args:
        cfg: config con num_classes y base_model

    Returns:
        modelo listo para fine-tuning
    """
    # TODO: cargar base preentrenado y reemplazar clasificador final
    # import timm
    # model = timm.create_model(cfg["base_model"], pretrained=True, num_classes=cfg["num_classes"])
    # return model
    pass


def cargar_datos(data_dir: str, cfg: dict):
    """Crea DataLoaders train/val.

    Args:
        data_dir: directorio con subcarpetas por clase
        cfg: config con image_size y batch_size

    Returns:
        (train_loader, val_loader)
    """
    # TODO: torchvision.datasets.ImageFolder + transformaciones + DataLoader
    # Estructura esperada:
    #   data_dir/
    #       train/manzana/ ... val/manzana/ ...
    pass


def entrenar(cfg: dict) -> None:
    """Loop de entrenamiento completo.

    Args:
        cfg: hiperparámetros
    """
    # TODO:
    # model = construir_modelo(cfg)
    # train_loader, val_loader = cargar_datos(cfg["data_dir"], cfg)
    # optimizer = Adam(model.parameters(), lr=cfg["learning_rate"])
    # criterion = CrossEntropyLoss()
    # for epoch in range(cfg["epochs"]):
    #     train_epoch(model, train_loader, optimizer, criterion)
    #     val_loss, val_acc = validate(model, val_loader, criterion)
    #     guardar_mejor(model, val_acc, cfg["output_dir"])
    pass


def guardar_mejor(model, mejor_acc: float, output_dir: str) -> None:
    """Guarda checkpoint si mejora la accuracy.

    Args:
        model: modelo pytorch
        mejor_acc: accuracy actual en validacion
        output_dir: donde guardar el .pt
    """
    # TODO: torch.save con logica de "es el mejor hasta ahora"
    pass


if __name__ == "__main__":
    cfg = cargar_config_entrenamiento()
    print(f"[Train Classifier] Config: {cfg}")
    entrenar(cfg)
