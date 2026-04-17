import base64
import cv2
import numpy as np
from detector_agent.tracker import TrackedObject


class Cropper:

    def __init__(self, padding=10):
        self.padding = padding

    def recortar(self, frame, obj):

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = obj.detection.bbox

        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)
        x2 = min(w, x2 + self.padding)
        y2 = min(h, y2 + self.padding)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return frame

        return crop

    def a_base64(self, img):

        ok, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])

        if not ok:
            raise RuntimeError("Error encoding image")

        return base64.b64encode(buffer).decode()

    def procesar(self, frame, obj):
        crop = self.recortar(frame, obj)
        return self.a_base64(crop)