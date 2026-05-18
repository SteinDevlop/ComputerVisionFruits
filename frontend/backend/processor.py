"""
frontend/backend/processor.py
Procesamiento de video en background thread.
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
            from detector_agent.detector import FruitDetector
            from detector_agent.tracker import FruitTracker
            from detector_agent.cropper import Cropper
            from detector_agent.client import ClassifierClient
            from detector_agent.visualizer import FrameVisualizer

            job["progress"] = 5
            detector = FruitDetector(conf_threshold=0.4, adaptive_confidence=True, preprocess_frames=True)
            detector.load_model()
            tracker = FruitTracker()
            cropper = Cropper()
            client = ClassifierClient()

            job["status"] = JobStatus.PROCESSING
            job["progress"] = 10

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
                    dets = detector.detect(frame)
                    objetos = tracker.update(dets)
                    alto, ancho = frame.shape[:2]
                    cx = ancho / 2
                    mx = ancho * 0.20

                    for obj in objetos:
                        if not obj.clasificado:
                            x1, y1, x2, y2 = obj.detection.bbox
                            if abs((x1 + x2) / 2 - cx) <= mx:
                                try:
                                    b64 = cropper.procesar(frame, obj)
                                    res = client.clasificar(obj.id_objeto, b64)
                                    if res:
                                        objects_info[obj.id_objeto] = {
                                            "fruta": res.fruta,
                                            "confianza": res.confianza,
                                            "precio": res.precio,
                                        }
                                        if res.fruta not in ["Unknown Label", "Unknown", "Desconocida"]:
                                            obj.clasificado = True
                                        existing = next((d for d in detections_log if d["fruta"] == res.fruta), None)
                                        if not existing:
                                            detections_log.append({
                                                "fruta": res.fruta,
                                                "confianza": res.confianza,
                                                "precio": res.precio,
                                            })
                                except Exception:
                                    pass

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
            # Los navegadores solo reproducen H.264 nativamente; mp4v (MPEG-4 Part 2)
            # requiere plugins o descarga completa del archivo.
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
                        "-movflags", "+faststart",  # moov al inicio → streaming inmediato
                        "-an",                       # sin audio (videos de detección no tienen)
                        h264_path,
                    ],
                    capture_output=True, timeout=600
                )
                if result.returncode == 0 and Path(h264_path).exists():
                    Path(output_path).unlink(missing_ok=True)
                    final_path = h264_path
                # Si ffmpeg falla, se usa el archivo original (descargable pero no streameable)
            except Exception:
                pass  # ffmpeg no disponible → usar mp4v original

            job["status"] = JobStatus.DONE
            job["progress"] = 100
            job["output_path"] = final_path
            job["detections"] = detections_log

        except Exception as e:
            job["status"] = JobStatus.ERROR
            job["error"] = str(e)
            job["traceback"] = traceback.format_exc()
