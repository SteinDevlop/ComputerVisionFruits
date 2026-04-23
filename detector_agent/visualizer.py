# detector_agent/visualizer.py
"""
Módulo visualizer: Dibujo de detecciones y guardado de video.

Dibuja bounding boxes, IDs y clasificaciones en frames,
muestra en ventana (si disponible) y guarda video de salida.
Compatible con Docker (headless mode).
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

from detector_agent.tracker import TrackedObject
from detector_agent.logger_config import setup_logger
from shared.config import detector_cfg, classifier_cfg

logger = setup_logger(__name__)


class FrameVisualizer:
    """
    Dibuja detecciones y clasificaciones en frames de video.
    
    Attributes:
        output_video_path: Ruta para guardar video de salida
        writer: VideoWriter de OpenCV
        display_available: True si hay display X11 disponible
        frame_count: Contador de frames guardados
    """

    def __init__(self, output_video_path: Optional[Path] = None):
        """
        Inicializa el visualizador.
        
        Args:
            output_video_path: Ruta donde guardar el video (opcional)
        """
        self.output_video_path = output_video_path
        self.writer: Optional[cv2.VideoWriter] = None
        self.display_available = self._check_display()
        self.frame_count = 0
        
        if not self.display_available:
            logger.info("No display disponible - ejecutándose en modo headless (Docker)")

    def _check_display(self) -> bool:
        """
        Verifica si hay display X11 disponible para GUI.
        """
        try:
            # En Docker headless, DISPLAY no estará disponible
            if os.environ.get('DISPLAY') is None:
                return False
            return True
        except:
            return False

    def initialize_writer(self, frame: np.ndarray, fps: float = 30.0) -> bool:
        """
        Inicializa el VideoWriter para guardar el video procesado.             
        Args:
            frame: Frame de referencia para obtener dimensiones
            fps: Frames por segundo del video de salida
            
        Returns:
            bool: True si es exitoso, False si no o no hay path"""
        try:
            if self.output_video_path is None:
                return False

            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            self.writer = cv2.VideoWriter(
                str(self.output_video_path),
                fourcc,
                fps,
                (width, height)
            )

            if not self.writer.isOpened():
                logger.error("No se pudo abrir VideoWriter")
                self.writer = None
                return False

            logger.info(f"VideoWriter inicializado: {self.output_video_path}")
            return True

        except Exception as e:
            logger.error(f"Error al inicializar writer: {e}")
            self.writer = None
            return False

    def draw_detections(
        self,
        frame: np.ndarray,
        objetos: List[TrackedObject],
        object_info: dict = None
    ) -> np.ndarray:
        """
        Dibuja cajas, IDs, clasificaciones y confianzas en el frame.
        Args:
            frame: Frame original
            objetos: Lista de objetos rastreados
            object_info: Dict {id_objeto: {"fruta": "...", "confianza": 0.95}}
            
        Returns:
            np.ndarray: Frame con detecciones dibujadas"""
        try:
            frame_vis = frame.copy()
            height, width = frame_vis.shape[:2]

            # Configuración de colores y tipografía
            COLOR_BOX = (0, 255, 0)  # Verde para cajas
            COLOR_UNKNOWN = (0, 165, 255)  # Naranja para no clasificado
            COLOR_INFO = (255, 255, 255)  # Blanco para texto
            THICKNESS = 2
            FONT = cv2.FONT_HERSHEY_SIMPLEX
            FONT_SCALE = 0.6
            FONT_THICKNESS = 1

            for obj in objetos:
                try:
                    x1, y1, x2, y2 = obj.detection.bbox
                    conf = obj.detection.confidence

                    # Validar coordenadas
                    x1 = max(0, min(int(x1), width - 1))
                    y1 = max(0, min(int(y1), height - 1))
                    x2 = max(x1 + 1, min(int(x2), width))
                    y2 = max(y1 + 1, min(int(y2), height))

                    # Seleccionar color según si está clasificado
                    if obj.clasificado and object_info and obj.id_objeto in object_info:
                        box_color = COLOR_BOX
                    else:
                        box_color = COLOR_UNKNOWN

                    # Dibujar caja
                    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), box_color, THICKNESS)

                    # Construir texto de info
                    info_lines = [f"ID: {obj.id_objeto}"]

                    if object_info and obj.id_objeto in object_info:
                        info = object_info[obj.id_objeto]
                        fruta = info.get("fruta", "Unknown")
                        confianza = info.get("confianza", 0.0)
                        info_lines.append(f"{fruta}")
                        info_lines.append(f"Conf: {confianza:.2%}")
                    else:
                        info_lines.append(f"Det: {conf:.2%}")

                    # Dibujar fondo para el texto
                    text_y_offset = y1 - 35
                    if text_y_offset < 20:
                        text_y_offset = y2 + 20

                    for i, text in enumerate(info_lines):
                        text_size = cv2.getTextSize(
                            text, FONT, FONT_SCALE, FONT_THICKNESS
                        )[0]

                        text_x = max(5, min(x1, width - text_size[0] - 5))
                        text_y = text_y_offset + (i * 20)

                        # Validar que el texto esté dentro del frame
                        if 0 <= text_y < height:
                            # Fondo negro para mejor legibilidad
                            cv2.rectangle(
                                frame_vis,
                                (text_x - 2, text_y - text_size[1] - 2),
                                (text_x + text_size[0] + 2, text_y + 2),
                                (0, 0, 0),
                                -1
                            )
                            # Texto blanco
                            cv2.putText(
                                frame_vis,
                                text,
                                (text_x, text_y),
                                FONT,
                                FONT_SCALE,
                                COLOR_INFO,
                                FONT_THICKNESS
                            )

                except Exception as e:
                    logger.debug(f"Error dibujando objeto {obj.id_objeto}: {e}")
                    continue

            return frame_vis

        except Exception as e:
            logger.error(f"Error en draw_detections: {e}")
            return frame

    def show_frame(self, frame: np.ndarray, window_name: str = "Detections") -> bool:
        """
        Muestra el frame en una ventana X11 (si disponible).
        
        Args:
            frame: Frame a mostrar
            window_name: Nombre de la ventana
            
        Returns:
            bool: True si se presionó 'Q', False en caso contrario"""
        try:
            if not self.display_available:
                return False

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            # Q para salir
            if key == ord('q'):
                return True

            return False

        except Exception as e:
            logger.debug(f"Error mostrando frame: {e}")
            return False

    def save_frame(self, frame: np.ndarray) -> bool:
        """
        Guarda el frame en el video de salida MP4.
        
        Args:
            frame: Frame a guardar
            
        Returns:
            bool: True si es exitoso, False si falla o writer no está disponible
        """
        try:
            if self.writer is None or not self.writer.isOpened():
                return False

            self.writer.write(frame)
            self.frame_count += 1
            return True

        except Exception as e:
            logger.error(f"Error guardando frame: {e}")
            return False

    def cleanup(self) -> None:
        """
        Libera recursos: cierra ventanas y escritor de video.
        
        Debe llamarse al final del procesamiento
        """
        try:
            if self.writer is not None and self.writer.isOpened():
                self.writer.release()
                logger.info(f"Video guardado: {self.output_video_path} ({self.frame_count} frames)")

            if self.display_available:
                cv2.destroyAllWindows()

        except Exception as e:
            logger.error(f"Error en cleanup: {e}")


def create_output_video_path(base_dir: Path = None) -> Path:
    """
    Genera un path único para el video de salida con timestamp.
    
    """
    try:
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent / "data"

        # Crear directorio si no existe
        base_dir.mkdir(parents=True, exist_ok=True)

        # Generar nombre con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = f"detection_output_{timestamp}.mp4"
        output_path = base_dir / video_name

        logger.debug(f"Path de salida generado: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error creando path de salida: {e}")
        return Path("data") / "detection_output.mp4"
