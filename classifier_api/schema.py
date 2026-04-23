# classifier_api/schema.py — modelos Pydantic entrada/salida

from pydantic import BaseModel, Field


class ClasificarRequest(BaseModel):
    """Payload que llega del detector."""
    id_objeto: int = Field(..., description="ID único del objeto trackeado")
    imagen: str = Field(..., description="Imagen recortada en base64")


class ClasificarResponse(BaseModel):
    """Respuesta del clasificador."""
    id_objeto: int
    fruta: str = Field(..., description="Clase predicha")
    confianza: float = Field(..., ge=0.0, le=1.0, description="Score de confianza 0-1")
    precio: int = Field(0, description="Precio de la fruta en COP")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
