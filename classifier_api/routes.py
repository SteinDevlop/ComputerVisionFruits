# classifier_api/routes.py — endpoints FastAPI

from fastapi import APIRouter, HTTPException

from classifier_api.inference import classifier
from classifier_api.schema import ClasificarRequest, ClasificarResponse, HealthResponse

router = APIRouter()


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

    return ClasificarResponse(
        id_objeto=payload.id_objeto,
        fruta=fruta,
        confianza=round(confianza, 4),
    )
