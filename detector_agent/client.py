# detector_agent/client.py — cliente HTTP hacia el clasificador

from dataclasses import dataclass

import httpx

from shared.config import api_cfg


@dataclass
class ResultadoClasificacion:
    id_objeto: int
    fruta: str
    confianza: float


class ClassifierClient:
    """Cliente HTTP para el Agente Clasificador."""

    def __init__(self, url: str | None = None, timeout: int | None = None):
        self.url = url or api_cfg.CLASSIFIER_URL
        self.timeout = timeout or api_cfg.TIMEOUT_SECONDS

    def clasificar(self, id_objeto: int, imagen_b64: str) -> ResultadoClasificacion | None:
        """Envía imagen al clasificador y retorna resultado.

        Args:
            id_objeto: ID del objeto trackeado
            imagen_b64: imagen recortada en base64

        Returns:
            ResultadoClasificacion o None si falla
        """
        payload = {"id_objeto": id_objeto, "imagen": imagen_b64}

        try:
            response = httpx.post(self.url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return ResultadoClasificacion(
                id_objeto=data["id_objeto"],
                fruta=data["fruta"],
                confianza=data["confianza"],
            )
        except httpx.TimeoutException:
            print(f"[Client] Timeout enviando objeto {id_objeto}")
            return None
        except httpx.HTTPError as e:
            print(f"[Client] HTTP error: {e}")
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
