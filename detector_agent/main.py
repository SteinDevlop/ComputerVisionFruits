# detector_agent/main.py — loop principal del agente detector

import cv2
import numpy as np

from detector_agent.client import ClassifierClient
from detector_agent.cropper import Cropper
from detector_agent.detector import FruitDetector
from detector_agent.tracker import FruitTracker
from shared.config import detector_cfg


def abrir_camara(source: int | str) -> cv2.VideoCapture:
    """Abre fuente de video.

    Args:
        source: 0 para cámara, o path a archivo de video

    Returns:
        VideoCapture listo para usar
    """
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, detector_cfg.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, detector_cfg.FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir fuente de video: {source}")
    return cap


def procesar_frame(
    frame: np.ndarray,
    detector: FruitDetector,
    tracker: FruitTracker,
    cropper: Cropper,
    client: ClassifierClient,
) -> None:
    """Pipeline completo para un frame.

    1. Detectar objetos con YOLO
    2. Asignar IDs con tracker
    3. Recortar cada objeto
    4. Enviar al clasificador
    5. Mostrar resultado (debug)

    Args:
        frame: frame BGR del video
        detector: agente YOLO
        tracker: gestor de IDs
        cropper: recortador de ROIs
        client: cliente HTTP clasificador
    """
    detecciones = detector.detect(frame)
    objetos = tracker.update(detecciones)

    for obj in objetos:
        imagen_b64 = cropper.procesar(frame, obj)
        resultado = client.clasificar(obj.id_objeto, imagen_b64)

        if resultado:
            # TODO: dibujar bbox + etiqueta en frame para visualizacion
            print(f"[Main] ID={resultado.id_objeto} | {resultado.fruta} | {resultado.confianza:.2f}")


def dibujar_debug(frame: np.ndarray, objetos: list, resultados: dict) -> np.ndarray:
    """Dibuja bboxes y labels en frame para visualizacion.

    Args:
        frame: frame original
        objetos: lista de TrackedObject
        resultados: dict {id_objeto: ResultadoClasificacion}

    Returns:
        frame con anotaciones
    """
    # TODO: cv2.rectangle + cv2.putText para cada objeto
    return frame


def main() -> None:
    """Entry point del agente detector. Loop principal."""
    print("[Main] Iniciando Agente Detector...")

    detector = FruitDetector()
    detector.load_model()

    tracker = FruitTracker()
    cropper = Cropper()
    client = ClassifierClient()

    cap = abrir_camara(detector_cfg.VIDEO_SOURCE)
    print(f"[Main] Cámara abierta: {detector_cfg.VIDEO_SOURCE}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Main] Sin frame — fin de video o error de cámara.")
                break

            procesar_frame(frame, detector, tracker, cropper, client)

            # TODO: mostrar frame con cv2.imshow si se quiere preview
            # cv2.imshow("FruitAI - Detector", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("[Main] Detenido por usuario.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.limpiar()
        print("[Main] Recursos liberados.")


if __name__ == "__main__":
    main()
