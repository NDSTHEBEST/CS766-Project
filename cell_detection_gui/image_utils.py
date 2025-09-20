"""Utility helpers for image loading, annotation, and metadata export."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .structures import Detection, DetectionSummary

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _load_grayscale_data(path: Path) -> Tuple[np.ndarray, Image.Image]:
    """Load an image as grayscale array and companion RGB image."""

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    with Image.open(path) as image:
        grayscale = image.convert("L")
        array = np.asarray(grayscale, dtype=np.uint8)
        display_image = grayscale.convert("RGB")
    return array, display_image


def create_image_stack(paths: Sequence[str]) -> Tuple[np.ndarray, Image.Image, List[str]]:
    """Create a three-channel stack from one to three grayscale images."""

    if not paths:
        raise ValueError("At least one image path must be provided.")

    selected_paths = [Path(p) for p in paths[:3]]
    arrays: List[np.ndarray] = []
    display_images: List[Image.Image] = []

    for path in selected_paths:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension for {path.name}.")
        array, display = _load_grayscale_data(path)
        arrays.append(array)
        display_images.append(display)

    reference_shape = arrays[0].shape
    for path, array in zip(selected_paths, arrays):
        if array.shape != reference_shape:
            raise ValueError(
                "All selected images must share the same dimensions. "
                f"Expected {reference_shape} but {path.name} has {array.shape}."
            )

    if len(arrays) == 1:
        prev_arr = curr_arr = next_arr = arrays[0]
    elif len(arrays) == 2:
        prev_arr, curr_arr = arrays
        next_arr = curr_arr
    else:
        prev_arr, curr_arr, next_arr = arrays[:3]

    stack = np.stack((prev_arr, curr_arr, next_arr), axis=-1)
    current_index = 1 if len(display_images) > 1 else 0
    base_image = display_images[current_index]

    return stack, base_image, [str(path) for path in selected_paths]


def prepare_detection_summary(
    image_size: Tuple[int, int],
    detection_boxes: np.ndarray,
    detection_scores: np.ndarray,
    threshold: float,
    source_paths: Iterable[str],
    model_info: dict,
) -> DetectionSummary:
    """Convert model outputs into a structured summary."""

    height, width = image_size
    detections: List[Detection] = []

    for box, score in zip(detection_boxes, detection_scores):
        if float(score) < threshold:
            continue
        ymin, xmin, ymax, xmax = box
        x0 = max(0, min(width, int(round(xmin * width))))
        y0 = max(0, min(height, int(round(ymin * height))))
        x1 = max(0, min(width, int(round(xmax * width))))
        y1 = max(0, min(height, int(round(ymax * height))))
        if x1 <= x0 or y1 <= y0:
            continue
        detection_id = len(detections) + 1
        detections.append(
            Detection(id=detection_id, score=float(score), bbox=(x0, y0, x1, y1))
        )

    return DetectionSummary(
        detections=detections,
        cell_count=len(detections),
        image_size=(width, height),
        detection_threshold=threshold,
        source_paths=[str(path) for path in source_paths],
        model_info=model_info,
    )


def draw_detections_on_image(image: Image.Image, summary: DetectionSummary) -> Image.Image:
    """Overlay bounding boxes and labels on an image."""

    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=14)
    except OSError:
        font = ImageFont.load_default()

    for detection in summary.detections:
        x0, y0, x1, y1 = detection.bbox
        draw.rectangle([(x0, y0), (x1, y1)], outline=(0, 255, 0), width=2)
        label = f"{detection.id}: {detection.score:.2f}"
        text_bbox = draw.textbbox((x0, y0), label, font=font)
        bg_coords = [
            (text_bbox[0] - 2, text_bbox[1] - 2),
            (text_bbox[2] + 2, text_bbox[3] + 2),
        ]
        draw.rectangle(bg_coords, fill=(0, 0, 0, 127))
        draw.text((x0, y0), label, fill=(255, 255, 255), font=font)

    return annotated


def build_metadata(summary: DetectionSummary) -> dict:
    """Generate a serializable metadata dictionary for detections."""

    width, height = summary.image_size
    metadata = {
        "model": summary.model_info,
        "detection_threshold": summary.detection_threshold,
        "image": {
            "width": width,
            "height": height,
            "source_files": list(summary.source_paths),
        },
        "cell_count": summary.cell_count,
        "detections": [],
    }

    for detection in summary.detections:
        x0, y0, x1, y1 = detection.bbox
        metadata["detections"].append(
            {
                "id": detection.id,
                "score": round(float(detection.score), 4),
                "bbox": {
                    "xmin": x0,
                    "ymin": y0,
                    "xmax": x1,
                    "ymax": y1,
                },
            }
        )

    return metadata


def save_metadata(metadata: dict, output_path: Path) -> None:
    """Persist detection metadata as JSON."""

    output_path = Path(output_path)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
