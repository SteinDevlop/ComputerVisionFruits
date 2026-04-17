import cv2
import numpy as np

from detector_agent.client import ClassifierClient
from detector_agent.cropper import Cropper
from detector_agent.detector import FruitDetector
from detector_agent.tracker import FruitTracker
from shared.config import detector_cfg


def abrir_video(source: int | str):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir video: {source}")

    return cap


def procesar_frame(frame, detector, tracker, cropper, client):

    detecciones = detector.detect(frame)
    objetos = tracker.update(detecciones)

    for obj in objetos:

        # 1. crop de la fruta
        imagen_b64 = cropper.procesar(frame, obj)

        # 2. envío a API clasificador
        resultado = client.clasificar(obj.id_objeto, imagen_b64)

        # 3. log simple
        if resultado:
            print(
                f"[API] ID={resultado.id_objeto} | "
                f"{resultado.fruta} | "
                f"{resultado.confianza:.2f}"
            )


def main():

    print("[Main] Detector → API pipeline")

    detector = FruitDetector()
    detector.load_model()

    tracker = FruitTracker()
    cropper = Cropper()
    client = ClassifierClient()

    cap = abrir_video(detector_cfg.VIDEO_SOURCE)

    frame_id = 0

    try:
        while True:

            ret, frame = cap.read()
            if not ret:
                print("[Main] Fin del video")
                break

            procesar_frame(frame, detector, tracker, cropper, client)

            frame_id += 1

    except KeyboardInterrupt:
        print("[Main] Detenido por usuario")

    finally:
        cap.release()
        tracker.limpiar()
        print("[Main] Recursos liberados")


if __name__ == "__main__":
    main()