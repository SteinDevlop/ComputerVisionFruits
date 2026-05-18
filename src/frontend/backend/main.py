"""
frontend/backend/main.py  —  Backend Gateway para FruitVision AI
Endpoints:
  GET  /api/health
  POST /api/detect-frame          ← NUEVO: cámara en vivo
  GET  /api/camera/reset          ← NUEVO: reiniciar tracker
  POST /api/process-video
  GET  /api/job/{job_id}
  GET  /api/job/{job_id}/stream
  GET  /api/job/{job_id}/download
  GET  /api/videos
  GET  /api/videos/{filename}
  WS   /ws/{job_id}
"""

import asyncio
import uuid
import os
import sys
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from frontend.backend.processor import VideoProcessor, jobs_store, JobStatus
from frontend.backend.camera_processor import (
    process_frame, reset_tracker, reset_cache, load_pipeline
)

app = FastAPI(title="FruitVision Gateway", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSIFIER_URL = os.getenv(
    "CLASSIFIER_URL",
    f"{os.getenv('CLASSIFIER_HOST', 'http://localhost')}:{os.getenv('CLASSIFIER_PORT', '8000')}/health"
)
# URL base del classifier para el health check
CLASSIFIER_HEALTH_URL = (
    f"{os.getenv('CLASSIFIER_HOST', 'http://localhost')}:{os.getenv('CLASSIFIER_PORT', '8000')}/health"
)


def _video_stream_response(path: str, request: Request, filename: str | None = None):
    """Sirve un video con soporte correcto de Range requests para HTML5."""
    file_path = Path(path)
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": "video/mp4",
    }
    if filename:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'

    if range_header:
        # Parsear Range: bytes=start-end
        range_val = range_header.strip().replace("bytes=", "")
        parts = range_val.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)
        content_length = end - start + 1

        def iter_file(s, e):
            with open(file_path, "rb") as f:
                f.seek(s)
                remaining = e - s + 1
                chunk = 65536  # 64KB chunks
                while remaining > 0:
                    data = f.read(min(chunk, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        headers["Content-Length"] = str(content_length)
        return StreamingResponse(
            iter_file(start, end), status_code=206, headers=headers
        )

    # Sin Range: enviar todo
    headers["Content-Length"] = str(file_size)
    return FileResponse(path, headers=headers, media_type="video/mp4")


# ── Health ─────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    classifier_ok = False
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(CLASSIFIER_HEALTH_URL)
            classifier_ok = r.status_code == 200
    except Exception:
        pass

    from frontend.backend.camera_processor import _loaded, _load_error
    return {
        "gateway": "ok",
        "classifier_api": "ok" if classifier_ok else "unavailable",
        "detector_loaded": _loaded,
        "detector_error": _load_error,
    }


# ── Cámara en vivo ──────────────────────────────────────────────────────
@app.post("/api/detect-frame")
async def detect_frame(payload: dict):
    """
    Recibe un frame base64, corre YOLO+Tracker+Classifier.
    Retorna lista de detecciones con bbox + clasificación.
    """
    imagen_b64 = payload.get("imagen")
    if not imagen_b64:
        raise HTTPException(400, "imagen requerida")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, process_frame, imagen_b64)
    return result


@app.post("/api/camera/reset")
async def camera_reset():
    """Reinicia el tracker y caché (nueva sesión de cámara)."""
    reset_tracker()
    reset_cache()
    return {"status": "ok", "message": "Tracker y caché reiniciados"}


@app.post("/api/camera/preload")
async def camera_preload():
    """Carga el modelo YOLO en background (puede tardar la primera vez)."""
    loop = asyncio.get_event_loop()
    ok, err = await loop.run_in_executor(None, load_pipeline)
    return {"loaded": ok, "error": err}


UPLOAD_DIR = PROJECT_ROOT / "frontend" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Cámara en vivo ──────────────────────────────────────────────────
@app.post("/api/process-video")
async def process_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        raise HTTPException(400, f"Formato no soportado: {ext}")

    job_id = str(uuid.uuid4())[:8]
    input_path = UPLOAD_DIR / f"{job_id}{ext}"
    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    jobs_store[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "progress": 0,
        "current_frame": 0,
        "total_frames": 0,
        "objects_classified": 0,
        "output_path": None,
        "error": None,
        "detections": [],
        "filename": file.filename,
    }
    asyncio.create_task(VideoProcessor.run(job_id, str(input_path)))
    return {"job_id": job_id, "status": "accepted"}


@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs_store:
        raise HTTPException(404, "Job no encontrado")
    return jobs_store[job_id]


@app.get("/api/job/{job_id}/stream")
async def stream_video(job_id: str, request: Request):
    job = jobs_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado")
    if job["status"] != JobStatus.DONE:
        raise HTTPException(400, "Video no listo aún")
    path = job.get("output_path")
    if not path or not Path(path).exists():
        raise HTTPException(404, "Archivo no encontrado")
    return _video_stream_response(path, request)


@app.get("/api/job/{job_id}/download")
async def download_video(job_id: str, request: Request):
    job = jobs_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado")
    if job["status"] != JobStatus.DONE:
        raise HTTPException(400, "Video no listo aún")
    path = job.get("output_path")
    if not path or not Path(path).exists():
        raise HTTPException(404, "Archivo no encontrado")
    return _video_stream_response(path, request, filename=f"detection_{job_id}.mp4")


# ── Videos guardados ───────────────────────────────────────────────────
@app.get("/api/videos")
async def list_videos():
    videos = []
    for p in sorted(OUTPUT_DIR.glob("detection_output_*.mp4"), reverse=True):
        s = p.stat()
        videos.append({
            "filename": p.name,
            "size_mb": round(s.st_size / 1_048_576, 2),
            "created_at": s.st_mtime,
        })
    return {"videos": videos}


@app.get("/api/videos/{filename}")
async def serve_video(filename: str, request: Request):
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "Nombre inválido")
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Video no encontrado")
    return _video_stream_response(str(path), request)


# ── WebSocket progreso ─────────────────────────────────────────────────
@app.websocket("/ws/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            if job_id not in jobs_store:
                await websocket.send_json({"error": "Job no encontrado"})
                break
            job = jobs_store[job_id]
            await websocket.send_json(job)
            if job["status"] in [JobStatus.DONE, JobStatus.ERROR]:
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
