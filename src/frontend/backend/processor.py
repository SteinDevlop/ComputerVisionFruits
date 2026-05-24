"""
frontend/backend/processor.py
Procesamiento de video en background thread.
Pipeline de 1 etapa: YOLOv8 unificado (best.pt)
"""
import asyncio
import sys
import traceback
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class JobStatus:
    PENDING = "pending"
    LOADING = "loading"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


jobs_store: dict = {}


class VideoProcessor:

    @staticmethod
    async def run(job_id: str, input_path: str):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, VideoProcessor._sync, job_id, input_path)

    @staticmethod
    def _sync(job_id: str, input_path: str):
        job = jobs_store[job_id]
        try:
            job["status"] = JobStatus.LOADING
            job["progress"] = 2

            import cv2
            import json
            from detector_agent.detector import FruitDetector
            from detector_agent.tracker import FruitTracker
            from detector_agent.visualizer import FrameVisualizer

            job["progress"] = 5

            # Etapa única: Detector YOLOv8 unificado
            detector = FruitDetector(
                conf_threshold=0.30,
                adaptive_confidence=True,
                preprocess_frames=True,
            )
            
            # Cargar el modelo best.pt unificado
            unified_model_path = PROJECT_ROOT / "backend" / "models" / "finetuned" / "best.pt"
            if not unified_model_path.exists():
                unified_model_path = PROJECT_ROOT / "models" / "finetuned" / "best.pt"
                
            detector.load_model(model_path=unified_model_path)

            tracker = FruitTracker()
            job["status"] = JobStatus.PROCESSING
            job["progress"] = 10

            # Cargar base de datos de precios
            precios_db = {}
            for candidate in [
                PROJECT_ROOT / "backend" / "database" / "precios.json",
                PROJECT_ROOT / "database" / "precios.json",
            ]:
                if candidate.exists():
                    with open(candidate, "r", encoding="utf-8") as f:
                        precios_db = json.load(f)
                    break

            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise RuntimeError(f"No se pudo abrir: {input_path}")

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            job["total_frames"] = total

            output_dir = PROJECT_ROOT / "data"
            output_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"detection_output_{ts}.mp4"

            visualizer = FrameVisualizer(output_video_path=output_path)
            objects_info = {}
            detections_log = []
            frame_id = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    # ── Etapa Única: Detectar y clasificar con YOLO ──
                    dets = detector.detect(frame)
                    objetos = tracker.update(dets)

                    for obj in objetos:
                        if not obj.clasificado:
                            conf = obj.detection.confidence
                            if conf < 0.30:
                                continue
                                
                            class_id = obj.detection.class_id
                            if hasattr(detector.model, 'names'):
                                yolo_label = detector.model.names[class_id]
                            else:
                                yolo_label = f"Clase_{class_id}"
                                
                            fruta = yolo_label.capitalize()
                            precio = precios_db.get(fruta, 0)

                            obj.clasificado = True
                            obj.etiqueta = fruta
                            obj.confianza = conf
                            obj.precio = precio

                            objects_info[obj.id_objeto] = {
                                "fruta": fruta,
                                "confianza": conf,
                                "precio": precio,
                            }

                            existing = next((d for d in detections_log if d["fruta"] == fruta), None)
                            if not existing:
                                detections_log.append({
                                    "fruta": fruta,
                                    "confianza": conf,
                                    "precio": precio,
                                })

                    if frame_id == 0 and visualizer.writer is None:
                        visualizer.initialize_writer(frame, fps=fps)

                    fv = visualizer.draw_detections(frame, objetos, objects_info)
                    visualizer.save_frame(fv)
                    frame_id += 1

                    if total > 0:
                        job["progress"] = min(10 + int((frame_id / total) * 85), 95)
                    job["current_frame"] = frame_id
                    job["objects_classified"] = len(objects_info)

                except Exception:
                    frame_id += 1
                    continue

            cap.release()
            visualizer.cleanup()

            # ── Convertir mp4v → H.264 para reproducción en navegador ──
            job["progress"] = 97
            job["status"] = "converting"
            final_path = str(output_path)
            try:
                import subprocess
                h264_path = str(output_path).replace(".mp4", "_web.mp4")
                result = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", str(output_path),
                        "-c:v", "libx264",
                        "-preset", "fast",
                        "-crf", "23",
                        "-movflags", "+faststart",
                        "-an",
                        h264_path,
                    ],
                    capture_output=True, timeout=600
                )
                if result.returncode == 0 and Path(h264_path).exists():
                    Path(output_path).unlink(missing_ok=True)
                    final_path = h264_path
            except Exception:
                pass

            job["status"] = JobStatus.DONE
            job["progress"] = 100
            job["output_path"] = final_path
            job["detections"] = detections_log

        except Exception as e:
            job["status"] = JobStatus.ERROR
            job["error"] = str(e)
            job["traceback"] = traceback.format_exc()
