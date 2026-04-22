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

    # Definir la zona central de la pantalla (un margen de 20% hacia los lados desde el centro)
    alto, ancho = frame.shape[:2]
    centro_x = ancho / 2
    margen_x = ancho * 0.20  

    for obj in objetos:

        if not obj.clasificado:
            # Calculamos dónde está la fruta en este cuadro
            x1, y1, x2, y2 = obj.detection.bbox
            fruta_centro_x = (x1 + x2) / 2

            # Verificamos si la fruta ya llegó a la zona central de la cámara
            if abs(fruta_centro_x - centro_x) <= margen_x:
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
                    
                    # 4. Marcamos como clasificado solo si la IA reconoció la fruta
                    # Si es desconocida, lo volverá a intentar en el siguiente frame
                    if resultado.fruta not in ["Unknown Label", "Unknown", "Desconocida"]:
                        obj.clasificado = True


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