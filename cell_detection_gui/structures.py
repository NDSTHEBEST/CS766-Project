"""Data structures used by the cell detection GUI application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Detection:
    """Represents a single detected cell."""

    id: int
    score: float
    bbox: Tuple[int, int, int, int]


@dataclass
class DetectionSummary:
    """Aggregated information about all detections in an image."""

    detections: List[Detection]
    cell_count: int
    image_size: Tuple[int, int]
    detection_threshold: float
    source_paths: List[str]
    model_info: Dict[str, str]
