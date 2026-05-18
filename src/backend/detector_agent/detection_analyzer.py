# detector_agent/detection_analyzer.py
"""
Módulo detection_analyzer: Análisis estadístico de detecciones.

Proporciona análisis estadístico de detecciones YOLO
sin sugerencias de umbrales ni preprocesamiento.
"""

import numpy as np
from typing import List, TYPE_CHECKING
from collections import defaultdict

from detector_agent.logger_config import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from detector_agent.detector import Detection


class DetectionAnalyzer:
    """
    Analiza estadísticas de detecciones YOLO para optimización.
    
    Registra historial de detecciones y proporciona:
    - Estadísticas de confianza (min, máx, promedio, mediana)
    - Tasa de frames sin detecciones
    - Sugerencias automáticas de umbrales óptimos
    - Reportes detallados
    """

    def __init__(self):
        """Inicializa el analizador con histórico vacío."""
        self.stats = {
            "total_detecciones": 0,
            "detecciones_por_threshold": defaultdict(int),
            "confianzas_promedio": [],
            "frames_sin_detecciones": 0,
            "frames_totales": 0,
        }
        self.detection_history = []
        logger.debug("DetectionAnalyzer inicializado")

    def track_frame(self, detecciones: List, frame_id: int) -> None:
        """
        Registra estadísticas de un frame procesado.
        
        Args:
            detecciones: Lista de detecciones del frame
            frame_id: ID del frame procesado
        """
        self.stats["frames_totales"] += 1
        self.stats["total_detecciones"] += len(detecciones)

        if len(detecciones) == 0:
            self.stats["frames_sin_detecciones"] += 1
        else:
            confianzas = [d.confidence for d in detecciones]
            avg_conf = np.mean(confianzas)
            self.stats["confianzas_promedio"].append(avg_conf)

            for det in detecciones:
                self.detection_history.append({
                    "frame_id": frame_id,
                    "confidence": det.confidence,
                    "bbox": det.bbox,
                    "class_id": det.class_id,
                })

    def get_stats(self) -> dict:
        """
        Retorna estadísticas consolidadas de detecciones.
        
        Returns:
            dict: Estadísticas incluyendo confianzas min/máx/promedio/mediana
        """
        if self.stats["frames_totales"] == 0:
            return {}

        promedio_detecciones = self.stats["total_detecciones"] / self.stats["frames_totales"]
        tasa_frames_sin_det = self.stats["frames_sin_detecciones"] / self.stats["frames_totales"]

        confianzas = []
        for item in self.detection_history:
            confianzas.append(item["confidence"])

        return {
            "frames_totales": self.stats["frames_totales"],
            "total_detecciones": self.stats["total_detecciones"],
            "promedio_detecciones_por_frame": promedio_detecciones,
            "frames_sin_detecciones": self.stats["frames_sin_detecciones"],
            "tasa_frames_sin_detecciones": tasa_frames_sin_det,
            "confianza_minima": min(confianzas) if confianzas else 0.0,
            "confianza_maxima": max(confianzas) if confianzas else 0.0,
            "confianza_promedio": np.mean(confianzas) if confianzas else 0.0,
            "confianza_mediana": np.median(confianzas) if confianzas else 0.0,
        }
    def print_report(self) -> None:
        """
        Imprime reporte detallado de estadísticas de detección en consola.
        
        Incluye frames procesados, detecciones totales, confianzas
        y tasa de frames sin detecciones.
        """
        stats = self.get_stats()

        if not stats:
            logger.info("Sin datos de detecciones para reportar")
            return

        print("\n" + "="*60)
        print("[REPORTE DE DETECCIONES]")
        print("="*60)
        print(f"Frames procesados:              {stats['frames_totales']}")
        print(f"Total de detecciones:           {stats['total_detecciones']}")
        print(f"Promedio por frame:             {stats['promedio_detecciones_por_frame']:.2f}")
        print(f"Frames sin detecciones:         {stats['frames_sin_detecciones']}")
        print(f"Tasa sin detecciones:           {stats['tasa_frames_sin_detecciones']:.1%}")
        print(f"\nRango de confianzas:")
        print(f"  Mínima:                       {stats['confianza_minima']:.3f}")
        print(f"  Máxima:                       {stats['confianza_maxima']:.3f}")
        print(f"  Promedio:                     {stats['confianza_promedio']:.3f}")
        print(f"  Mediana:                      {stats['confianza_mediana']:.3f}")
        print("="*60 + "\n")

