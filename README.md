# 🍎 Fruit AI System

Sistema multi-agente para detección y clasificación de frutas en tiempo real.

---

## Arquitectura

```
Cámara → Detector (YOLO) → Tracker → Cropper → POST /clasificar → Clasificador → Resultado
```

Dos servicios independientes:
- **detector_agent** — lee video, detecta, trackea, envía crops
- **classifier_api** — recibe imagen base64, retorna tipo de fruta + confianza

---

## Estructura

```
fruit-ai-system/
├── data/training/
│   ├── detector/
│   │   ├── train.py        # fine-tuning YOLO
│   │   └── dataset.py      # dataset.yaml + utilidades
│   └── classifier/
│       ├── train.py        # fine-tuning clasificador
│       └── dataset.py      # FruitDataset + transforms
│
├── models/
│   ├── pretrained/         # poner aqui modelos base
│   └── finetuned/          # detector_best.pt + classifier_best.pt
│
├── detector_agent/
│   ├── main.py             # loop principal
│   ├── detector.py         # wrapper YOLO
│   ├── tracker.py          # asignacion de IDs
│   ├── cropper.py          # recorte + base64
│   └── client.py           # HTTP client → /clasificar
│
├── classifier_api/
│   ├── main.py             # FastAPI app
│   ├── routes.py           # GET /health, POST /clasificar
│   ├── inference.py        # FruitClassifier singleton
│   └── schema.py           # Pydantic models
│
├── shared/
│   └── config.py           # configuracion centralizada
│
└── deployment/
    ├── docker-compose.yml
    ├── Dockerfile.classifier
    ├── Dockerfile.detector
    ├── requirements.classifier.txt
    ├── requirements.detector.txt
    └── requirements.training.txt
```

---

## Quickstart

### 1. Entrenar modelos

```bash
pip install -r deployment/requirements.training.txt

# Detector
python -m data.training.detector.train

# Clasificador
python -m data.training.classifier.train
```

Modelos se guardan en `models/finetuned/`.

### 2. Levantar con Docker

```bash
cd deployment
docker-compose up --build
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
