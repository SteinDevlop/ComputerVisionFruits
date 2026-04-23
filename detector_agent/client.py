# detector_agent/client.py
"""
Módulo client: Cliente HTTP para comunicación con API clasificador.

Envía imágenes de frutas recortadas al API clasificador
y recibe predicciones de clasificación.
"""

from dataclasses import dataclass

import httpx

from shared.config import api_cfg
from detector_agent.logger_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class ResultadoClasificacion:
    """
    Resultado de clasificación retornado por el API.
    
    Attributes:
        id_objeto: ID del objeto rastreado
        fruta: Nombre de la fruta clasificada
        confianza: Confianza de la clasificación (0.0 a 1.0)
    """
    id_objeto: int
    fruta: str
    confianza: float


class ClassifierClient:
    """
    Cliente HTTP para el API clasificador de frutas.
    
    Realiza solicitudes POST al clasificador con imágenes en base64
    y parsea las respuestas JSON.
    """

    def __init__(self, url: str | None = None, timeout: int | None = None):
        """
        Inicializa el cliente clasificador.
        
        Args:
            url: URL del endpoint clasificador. Si es None, usa API_CFG
            timeout: Timeout en segundos. Si es None, usa API_CFG
        """
        self.url = url or api_cfg.CLASSIFIER_URL
        self.timeout = timeout or api_cfg.TIMEOUT_SECONDS
        logger.info(f"Cliente clasificador inicializado: {self.url}")

    def clasificar(self, id_objeto: int, imagen_b64: str) -> ResultadoClasificacion | None:
        """
        Envía una imagen al clasificador y obtiene su predicción.
        
        Realiza una solicitud POST HTTP con payload JSON conteniendo
        el ID del objeto y la imagen codificada en base64.
        
        Args:
            id_objeto: ID único del objeto rastreado
            imagen_b64: Imagen del objeto codificada en base64
            
        Returns:
            ResultadoClasificacion si es exitoso, None si falla
            
        Raises:
            Captura excepciones de HTTP y timeout silenciosamente
        """
        payload = {"id_objeto": id_objeto, "imagen": imagen_b64}

        try:
            response = httpx.post(self.url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            resultado = ResultadoClasificacion(
                id_objeto=data["id_objeto"],
                fruta=data["fruta"],
                confianza=data["confianza"],
            )
            logger.debug(f"Clasificación exitosa: {resultado}")
            return resultado
            
        except httpx.TimeoutException:
            logger.warning(f"Timeout enviando objeto {id_objeto}")
            return None
        except httpx.HTTPError as e:
            logger.error(f"Error HTTP: {e}")
            return None
        except Exception as e:
            logger.error(f"Error inesperado clasificando: {e}")
            return None

    async def clasificar_async(self, id_objeto: int, imagen_b64: str) -> ResultadoClasificacion | None:
        """Version async del clasificar. Usar si el loop lo permite.

        Args:
            id_objeto: ID del objeto trackeado
            imagen_b64: imagen base64

        Returns:
            ResultadoClasificacion o None si falla
        """
        payload = {"id_objeto": id_objeto, "imagen": imagen_b64}

        # TODO: usar en contexto async con httpx.AsyncClient
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                return ResultadoClasificacion(**data)
            except Exception as e:
                print(f"[Client] Error async: {e}")
                return None
