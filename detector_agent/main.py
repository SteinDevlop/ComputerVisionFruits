import cv2
import numpy as np
from pathlib import Path

from detector_agent.client import ClassifierClient
from detector_agent.cropper import Cropper
from detector_agent.detector import FruitDetector
from detector_agent.tracker import FruitTracker
from detector_agent.visualizer import FrameVisualizer, create_output_video_path
from detector_agent.detection_analyzer import DetectionAnalyzer
from shared.config import detector_cfg

def abrir_video(source: int | str):
    """Abre una fuente de video (cámara o archivo)"""
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir video: {source}")

    return cap


def obtener_fps(cap: cv2.VideoCapture) -> float:
    """Obtiene los FPS del video"""
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0
    except:
        return 30.0


def procesar_frame(frame, detector, tracker, cropper, client, object_info, analyzer):
    """
    Procesa un frame: detecta, rastrea y clasifica frutas.
    Actualiza object_info con clasificaciones.
    Registra estadísticas en analyzer.
    """
    try:
        detecciones = detector.detect(frame)
        
        # Registrar en analizador
        analyzer.track_frame(detecciones, frame_id=0)  # frame_id será actualizado en main
        
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
                    try:
                        # 1. crop de la fruta
                        imagen_b64 = cropper.procesar(frame, obj)

                        # 2. envío a API clasificador
                        resultado = client.clasificar(obj.id_objeto, imagen_b64)

                        # 3. log y actualizar info
                        if resultado:
                            print(
                                f"[API] ID={resultado.id_objeto} | "
                                f"{resultado.fruta} | "
                                f"{resultado.confianza:.2f} | "
                                f"${resultado.precio} COP"
                            )

                            # Almacenar clasificación
                            object_info[obj.id_objeto] = {
                                "fruta": resultado.fruta,
                                "confianza": resultado.confianza,
                                "precio": resultado.precio
                            }

                            # 4. Marcamos como clasificado solo si la IA reconoció la fruta
                            if resultado.fruta not in ["Unknown Label", "Unknown", "Desconocida"]:
                                obj.clasificado = True

                    except Exception as e:
                        print(f"[Main] Error clasificando objeto {obj.id_objeto}: {e}")
                        continue

        return objetos

    except Exception as e:
        print(f"[Main] Error procesando frame: {e}")
        return []


def main():
    """
    Pipeline completo: detección, rastreo, clasificación y visualización.
    Anti-fallo: captura excepciones y limpia recursos correctamente.
    Incluye análisis de detecciones para optimización.
    """

    print("[Main] ===== DETECTOR ROBUSTO CON ANÁLISIS =====")
    print("[Main] Iniciando pipeline: Detección → Rastreo → Clasificación → Visualización")

    detector = None
    tracker = None
    cropper = None
    client = None
    cap = None
    visualizer = None
    analyzer = None

    try:
        # Inicializar componentes
        print("\n[Main] Inicializando componentes...")

        detector = FruitDetector(
            conf_threshold=0.4,
            adaptive_confidence=True,
            preprocess_frames=True,
            multi_scale=False
        )
        detector.load_model()

        tracker = FruitTracker()
        cropper = Cropper()
        client = ClassifierClient()
        analyzer = DetectionAnalyzer()

        # Abrir video
        print(f"[Main] Abriendo video: {detector_cfg.VIDEO_SOURCE}")
        cap = abrir_video(detector_cfg.VIDEO_SOURCE)
        fps = obtener_fps(cap)
        print(f"[Main] FPS del video: {fps}")

        # Crear visualizador y video de salida
        output_path = create_output_video_path()
        visualizer = FrameVisualizer(output_video_path=output_path)

        frame_id = 0
        should_quit = False
        objects_info = {}  # {id_objeto: {"fruta": "...", "confianza": 0.95}}

        print("\n[Main] Procesando video...")
        print("[Main] (Presionar 'Q' en la ventana para salir)\n")

        while not should_quit:

            ret, frame = cap.read()
            if not ret:
                print("[Main] Fin del video - todos los frames procesados")
                break

            try:
                # Procesar frame
                objetos = procesar_frame(
                    frame, detector, tracker, cropper, client, objects_info, analyzer
                )

                # Inicializar writer en el primer frame
                if frame_id == 0 and visualizer.writer is None:
                    visualizer.initialize_writer(frame, fps=fps)

                # Dibujar detecciones y clasificaciones
                frame_vis = visualizer.draw_detections(frame, objetos, objects_info)

                # Mostrar en ventana (si disponible)
                if visualizer.show_frame(frame_vis):
                    should_quit = True

                # Guardar en video
                visualizer.save_frame(frame_vis)

                frame_id += 1

                # Log de progreso cada 30 frames
                if frame_id % 30 == 0:
                    current_threshold = detector.get_current_threshold()
                    print(
                        f"[Main] Frame {frame_id} | "
                        f"Objetos: {len(objetos)} | "
                        f"Clasificados: {len(objects_info)} | "
                        f"Threshold: {current_threshold:.3f}"
                    )

            except Exception as e:
                print(f"[Main] Error procesando frame {frame_id}: {e}")
                continue

    except KeyboardInterrupt:
        print("\n[Main] Detenido por usuario")

    except Exception as e:
        print(f"[Main] Error fatal: {e}")

    finally:
        # Limpiar recursos
        print("\n[Main] Limpiando recursos...")

        try:
            if cap is not None:
                cap.release()
        except:
            pass

        try:
            if tracker is not None:
                tracker.limpiar()
        except:
            pass

        try:
            if visualizer is not None:
                visualizer.cleanup()
        except:
            pass

        # Mostrar análisis de detecciones
        try:
            print("\n[Main] Analizando resultados...")
            if analyzer is not None:
                analyzer.print_report()
        except Exception as e:
            print(f"[Main] Error mostrando análisis: {e}")

        print("[Main] Proceso finalizado exitosamente")
        print(f"[Main] Total objetos clasificados: {len(objects_info)}")
        print("[Main] ================================================\n")


if __name__ == "__main__":
    main()