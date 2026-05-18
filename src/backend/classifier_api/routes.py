# classifier_api/routes.py — endpoints FastAPI

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

from classifier_api.inference import classifier
from classifier_api.schema import ClasificarRequest, ClasificarResponse, HealthResponse

router = APIRouter()

# Cargar base de datos de precios
precios_json_path = Path(__file__).resolve().parent.parent / "database" / "precios.json"
precios_db = {}
if precios_json_path.exists():
    with open(precios_json_path, "r", encoding="utf-8") as f:
        precios_db = json.load(f)

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Verifica que la API y el modelo estén listos."""
    return HealthResponse(
        status="ok",
        model_loaded=classifier.is_loaded,
    )


@router.post("/clasificar", response_model=ClasificarResponse)
async def clasificar(payload: ClasificarRequest):
    """Recibe imagen base64 y retorna clase + confianza.

    Args:
        payload: { id_objeto: int, imagen: base64_str }

    Returns:
        { id_objeto, fruta, confianza }
    """
    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    try:
        fruta, confianza = classifier.predict(payload.imagen)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {e}")

    if confianza < 0.7:
        fruta = "Desconocida"
        precio = 0
    else:
        # Buscar el precio en el diccionario cargado del json
        precio = precios_db.get(fruta, 0)
        
    return ClasificarResponse(
        id_objeto=payload.id_objeto,
        fruta=fruta,
        confianza=round(confianza, 4),
        precio=precio
    )
