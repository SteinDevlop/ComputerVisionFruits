# 🍎 Fruit AI System

Sistema de detección y clasificación de frutas en tiempo real.

---

## Arquitectura

```
Cámara / Video → Gateway (FastAPI + React) → Classifier API → Resultado
```

Servicios en despliegue:
- **gateway** — frontend + backend web que orquesta la carga de video, la detección y las llamadas al clasificador
- **classifier_api** — servicio FastAPI que recibe imagen base64 y retorna fruta + confianza

> Nota: `detector_agent` ya no se despliega como servicio independiente; su lógica de desarrollo permanece en `src/backend/detector_agent`.

---

## Estructura

```
fruit-ai-system/
├── src/backend/
│   ├── classifier_api/
│   │   ├── main.py             # FastAPI app
│   │   ├── routes.py           # GET /health, POST /clasificar
│   │   ├── inference.py        # FruitClassifier singleton
│   │   └── schema.py           # Pydantic models
│   ├── data/training/
│   │   ├── detector/
│   │   │   ├── train.py        # fine-tuning YOLO
│   │   │   └── dataset.py      # dataset.yaml + utilidades
│   │   └── classifier/
│   │       ├── train.py        # fine-tuning clasificador
│   │       └── dataset.py      # FruitDataset + transforms
│   ├── detector_agent/
│   │   ├── cropper.py          # recorte + base64
│   │   ├── detector.py         # wrapper YOLO
│   │   ├── logger_config.py
│   │   ├── tracker.py          # asignacion de IDs
│   │   └── client.py           # HTTP client → /clasificar
│   ├── database/
│   ├── models/
│   │   ├── pretrained/         # poner aqui modelos base
│   │   └── finetuned/          # detector_best.pt + classifier_best.pt
│   └── shared/
│       └── config.py           # configuracion centralizada
├── src/frontend/
│   ├── backend/
│   │   ├── main.py
│   │   ├── camera_processor.py
│   │   ├── processor.py
│   │   ├── requirements.txt
│   │   └── __init__.py
│   ├── uploads/
│   ├── Dockerfile
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── deployment/
    ├── docker-compose.yml
    ├── Dockerfile.classifier
    ├── Dockerfile.gateway
    ├── nginx.conf
    ├── requirements.classifier.txt
    └── requirements.training.txt
```

---

## Quickstart

### 1. Entrenar modelos

```bash
cd src/backend
pip install -r ../deployment/requirements.training.txt

# Detector
python -m data.training.detector.train

# Clasificador
python -m data.training.classifier.train
```

Modelos se guardan en `src/backend/models/finetuned/`.

### 2. Levantar con Docker

```bash
cd deployment
docker compose up --build
```

### 3. Test manual del clasificador

```bash
curl -X POST http://localhost:8000/clasificar \
  -H "Content-Type: application/json" \
  -d '{"id_objeto": 1, "imagen": "<base64_string>"}'
```

Respuesta esperada:
```json
{
  "id_objeto": 1,
  "fruta": "manzana",
  "confianza": 0.97
}
```

---

## API Reference

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/health` | GET | Estado del servicio y modelo |
| `/clasificar` | POST | Clasifica imagen base64 |

---

## TODOs principales

- [ ] `detector.py` — implementar `detect()` con ultralytics YOLO
- [ ] `tracker.py` — implementar `_calcular_iou()` y `update()`
- [ ] `inference.py` — implementar `load_model()` y `predict()`
- [ ] `data/training/*/train.py` — loops de entrenamiento reales
- [ ] `data/training/*/dataset.py` — carga de datos reales
- [ ] `shared/config.py` — ajustar clases y rutas al dataset real

---

## Configuración

Editar `shared/config.py` para cambiar:
- Fuente de video (`VIDEO_SOURCE`)
- Umbrales de detección (`CONFIDENCE_THRESHOLD`)
- URL del clasificador (`CLASSIFIER_URL`)
- Clases de frutas (`CLASSES`)
