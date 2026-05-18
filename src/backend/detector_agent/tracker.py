from dataclasses import dataclass
from detector_agent.detector import Detection
from detector_agent.logger_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class TrackedObject:
    """
    Representa un objeto rastreado a lo largo del video.
    
    Attributes:
        id_objeto: Identificador único del objeto
        detection: Última detección del objeto
        frames_sin_ver: Contador de frames donde no se vio el objeto
        clasificado: Flag indicando si ya fue clasificado
        etiqueta: Etiqueta de la clasificación
        confianza: Score de confianza
        precio: Precio de la fruta
    """
    id_objeto: int
    detection: Detection
    frames_sin_ver: int = 0
    clasificado: bool = False
    etiqueta: str = ""
    confianza: float = 0.0
    precio: int = 0


class FruitTracker:
    """
    Rastreador de frutas usando IoU (Intersection over Union).
    
    Asigna IDs únicos persistent y rastrea objetos a través de frames.
    Utiliza IoU para asociar detecciones con objetos conocidos.
    
    Attributes:
        iou_threshold: Umbral mínimo de IoU para emparejar objetos
        max_frames_perdido: Máximo de frames sin ver un objeto antes de eliminarlo
    """

    def __init__(self, iou_threshold=0.3, max_frames_perdido=5):
        """
        Inicializa el rastreador.
        
        Args:
            iou_threshold: Umbral mínimo de IoU para asociar detecciones (0.0-1.0)
            max_frames_perdido: Máximo de frames sin detección antes de eliminar objeto
        """
        self.iou_threshold = iou_threshold
        self.max_frames_perdido = max_frames_perdido
        self._next_id = 0
        self._objetos = {}
        logger.debug(f"Tracker inicializado: IoU={iou_threshold}, max_frames={max_frames_perdido}")

    def _iou(self, a, b):
        """
        Calcula el Intersection over Union de dos bounding boxes.
        
        Args:
            a: Bounding box (x1, y1, x2, y2)
            b: Bounding box (x1, y1, x2, y2)
            
        Returns:
            float: IoU (0.0 a 1.0)
        """
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)

        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])

        return inter / (area_a + area_b - inter + 1e-6)

    def update(self, detecciones):
        """
        Actualiza el rastreador con nuevas detecciones.
        
        Proceso:
        1. Para cada detección, busca el objeto mejor emparejado por IoU
        2. Si IoU > threshold, actualiza el objeto existente
        3. Si no, crea un nuevo objeto con nuevo ID
        4. Marca como perdidos objetos no detectados
        5. Elimina objetos que han estado perdidos demasiado tiempo
        
        Args:
            detecciones: Lista de detecciones del frame actual
            
        Returns:
            list: Lista de objetos rastreados (TrackedObject)
        """
        new_objects = {}
        used = set()

        for det in detecciones:
            best_id = None
            best_iou = 0

            for obj_id, obj in self._objetos.items():
                iou = self._iou(det.bbox, obj.detection.bbox)

                if iou > best_iou:
                    best_iou = iou
                    best_id = obj_id

            if best_id is not None and best_iou > self.iou_threshold:
                obj = self._objetos[best_id]
                obj.detection = det
                obj.frames_sin_ver = 0
                new_objects[best_id] = obj
                used.add(best_id)
            else:
                new_id = self._next_id
                self._next_id += 1

                new_objects[new_id] = TrackedObject(
                    id_objeto=new_id,
                    detection=det
                )
                logger.debug(f"Nuevo objeto rastreado: ID={new_id}")

        for obj_id, obj in self._objetos.items():
            if obj_id not in used:
                obj.frames_sin_ver += 1
                if obj.frames_sin_ver < self.max_frames_perdido:
                    new_objects[obj_id] = obj
                else:
                    logger.debug(f"Objeto eliminado por inactividad: ID={obj_id}")

        self._objetos = new_objects
        return list(self._objetos.values())

    def limpiar(self):
        """
        Limpia el estado del rastreador.
        
        Elimina todos los objetos rastreados y resetea contadores.
        """
        self._objetos.clear()
        self._next_id = 0
        logger.debug("Rastreador limpiado")