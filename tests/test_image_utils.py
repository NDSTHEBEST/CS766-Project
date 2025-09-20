"""Unit tests for image utility helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from cell_detection_gui import image_utils
from cell_detection_gui.structures import Detection, DetectionSummary


def create_temp_image(directory: Path, name: str, value: int = 128) -> Path:
    path = directory / name
    image = Image.new("L", (32, 32), color=value)
    image.save(path)
    return path


def test_create_image_stack_single_image(tmp_path):
    img_path = create_temp_image(tmp_path, "slice.png", value=100)
    stack, base_image, used_paths = image_utils.create_image_stack([str(img_path)])

    assert stack.shape == (32, 32, 3)
    assert np.array_equal(stack[..., 0], stack[..., 1])
    assert np.array_equal(stack[..., 0], stack[..., 2])
    assert used_paths == [str(img_path)]
    assert base_image.size == (32, 32)


def test_create_image_stack_three_images(tmp_path):
    img1 = create_temp_image(tmp_path, "prev.png", value=70)
    img2 = create_temp_image(tmp_path, "current.png", value=120)
    img3 = create_temp_image(tmp_path, "next.png", value=200)

    stack, base_image, used_paths = image_utils.create_image_stack(
        [str(img1), str(img2), str(img3)]
    )

    assert stack.shape == (32, 32, 3)
    with Image.open(img1) as prev:
        prev_array = np.array(prev.convert("L"))
    with Image.open(img2) as current:
        current_array = np.array(current.convert("L"))
    with Image.open(img3) as nxt:
        next_array = np.array(nxt.convert("L"))

    assert np.array_equal(stack[..., 0], prev_array)
    assert np.array_equal(stack[..., 1], current_array)
    assert np.array_equal(stack[..., 2], next_array)
    assert used_paths == [str(img1), str(img2), str(img3)]
    assert base_image.size == (32, 32)


def test_prepare_detection_summary_filters_by_threshold(tmp_path):
    image_size = (100, 120)  # height, width
    boxes = np.array(
        [
            [0.1, 0.1, 0.3, 0.3],  # score below threshold
            [0.25, 0.25, 0.75, 0.75],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.4, 0.85])
    summary = image_utils.prepare_detection_summary(
        image_size=image_size,
        detection_boxes=boxes,
        detection_scores=scores,
        threshold=0.5,
        source_paths=["slice.png"],
        model_info={"name": "test-model"},
    )

    assert summary.cell_count == 1
    assert summary.detections[0].bbox == (30, 25, 90, 75)
    assert summary.source_paths == ["slice.png"]
    assert summary.model_info["name"] == "test-model"


def test_build_metadata_structure(tmp_path):
    summary = DetectionSummary(
        detections=[Detection(id=1, score=0.9, bbox=(30, 25, 90, 75))],
        cell_count=1,
        image_size=(120, 100),
        detection_threshold=0.5,
        source_paths=["slice.png"],
        model_info={"name": "test-model"},
    )
    metadata = image_utils.build_metadata(summary)

    assert metadata["cell_count"] == 1
    assert metadata["detections"][0]["bbox"]["xmin"] == 30
    assert metadata["model"]["name"] == "test-model"


def test_draw_detections_on_image_preserves_size():
    summary = DetectionSummary(
        detections=[Detection(id=1, score=0.9, bbox=(5, 5, 20, 20))],
        cell_count=1,
        image_size=(40, 40),
        detection_threshold=0.5,
        source_paths=["slice.png"],
        model_info={"name": "test-model"},
    )
    image = Image.new("RGB", (40, 40), color="black")
    annotated = image_utils.draw_detections_on_image(image, summary)

    assert annotated.size == image.size
    assert annotated.getpixel((5, 5)) != (0, 0, 0)
