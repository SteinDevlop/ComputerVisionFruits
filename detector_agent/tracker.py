from dataclasses import dataclass
from detector_agent.detector import Detection


@dataclass
class TrackedObject:
    id_objeto: int
    detection: Detection
    frames_sin_ver: int = 0
    clasificado: bool = False


class FruitTracker:

    def __init__(self, iou_threshold=0.3, max_frames_perdido=5):
        self.iou_threshold = iou_threshold
        self.max_frames_perdido = max_frames_perdido
        self._next_id = 0
        self._objetos = {}

    def _iou(self, a, b):
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

        for obj_id, obj in self._objetos.items():
            if obj_id not in used:
                obj.frames_sin_ver += 1
                if obj.frames_sin_ver < self.max_frames_perdido:
                    new_objects[obj_id] = obj

        self._objetos = new_objects
        return list(self._objetos.values())