# classifier_api/main.py — app FastAPI principal

from contextlib import asynccontextmanager

from fastapi import FastAPI

from classifier_api.inference import classifier
from classifier_api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga modelo al arrancar. Limpia al apagar."""
    classifier.load_model()
    yield
    # TODO: liberar recursos GPU si aplica


app = FastAPI(
    title="Fruit Classifier API",
    description="Recibe imagen base64, retorna tipo de fruta y confianza.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


# Ejecutar directo: uvicorn classifier_api.main:app --host 0.0.0.0 --port 8000
