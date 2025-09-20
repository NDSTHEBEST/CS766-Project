"""TensorFlow model loader for cell detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class _Backend:
    """Holds lazy-loaded TensorFlow objects."""

    tf: object
    detect_fn: object


class TensorFlowCellDetector:
    """Wraps the TensorFlow Object Detection pipeline used in the project."""

    def __init__(
        self,
        pipeline_config_path: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_prefix: str = "ckpt-10",
    ) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        default_checkpoint_dir = repo_root / "Machine Learning" / "Cell Counting Multichannel" / "checkpoint"
        default_pipeline_config = default_checkpoint_dir / "new_config" / "pipeline.config"

        self.pipeline_config_path = Path(pipeline_config_path or default_pipeline_config)
        self.checkpoint_dir = Path(checkpoint_dir or default_checkpoint_dir)
        self.checkpoint_prefix = checkpoint_prefix
        self._backend: Optional[_Backend] = None
        self._model_info: Dict[str, str] = {
            "name": "SSD ResNet50 V1 FPN (multichannel)",
            "pipeline_config": str(self.pipeline_config_path),
            "checkpoint": str(self.checkpoint_dir / self.checkpoint_prefix),
            "framework": "TensorFlow Object Detection API",
            "training_data": "Three-slice multichannel microscope stacks",
        }

    @property
    def model_info(self) -> Dict[str, str]:
        """Information describing the loaded model."""

        return dict(self._model_info)

    def _ensure_checkpoint_unpacked(self) -> None:
        """Extract the checkpoint if only the split archives are present."""

        index_path = self.checkpoint_dir / f"{self.checkpoint_prefix}.index"
        data_exists = any(
            self.checkpoint_dir.glob(f"{self.checkpoint_prefix}.data-*-of-*")
        )
        if index_path.exists() and data_exists:
            return

        first_volume = self.checkpoint_dir / f"{self.checkpoint_prefix}.7z.001"
        if not first_volume.exists():
            raise FileNotFoundError(
                "Compressed checkpoint archive not found. Expected "
                f"{first_volume.name} in {self.checkpoint_dir}."
            )

        try:
            from multivolumefile import MultiVolume
            import py7zr
        except ImportError as exc:
            raise ImportError(
                "py7zr is required to unpack the TensorFlow checkpoint archives. "
                "Install py7zr to continue."
            ) from exc

        base_archive = self.checkpoint_dir / f"{self.checkpoint_prefix}.7z"
        with MultiVolume(str(base_archive), mode="rb") as archive:
            with py7zr.SevenZipFile(archive, mode="r") as zf:
                zf.extractall(path=self.checkpoint_dir)

    def _load_backend(self) -> _Backend:
        if self._backend is not None:
            return self._backend

        try:
            import tensorflow as tf
            from object_detection.builders import model_builder
            from object_detection.utils import config_util
        except ImportError as exc:
            raise ImportError(
                "TensorFlow and the TensorFlow Object Detection API are required. "
                "Follow the setup instructions in docs/index.md to install them."
            ) from exc

        self._ensure_checkpoint_unpacked()
        configs = config_util.get_configs_from_pipeline_file(str(self.pipeline_config_path))
        model_config = configs["model"]
        detection_model = model_builder.build(model_config=model_config, is_training=False)
        checkpoint = tf.compat.v2.train.Checkpoint(model=detection_model)
        checkpoint_path = self.checkpoint_dir / self.checkpoint_prefix
        checkpoint.restore(str(checkpoint_path)).expect_partial()

        @tf.function
        def detect_fn(image):
            preprocessed_image, shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(preprocessed_image, shapes)
            return detection_model.postprocess(prediction_dict, shapes)

        self._backend = _Backend(tf=tf, detect_fn=detect_fn)
        return self._backend

    def predict(self, image_stack: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on a pre-processed three-channel image stack."""

        if image_stack.ndim != 3 or image_stack.shape[-1] != 3:
            raise ValueError(
                "image_stack must be a 3-D array with three channels (previous/current/next)."
            )

        backend = self._load_backend()
        tf = backend.tf
        detect_fn = backend.detect_fn

        array = np.asarray(image_stack, dtype=np.float32)
        input_tensor = tf.convert_to_tensor(np.expand_dims(array, axis=0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        return {
            "boxes": detections["detection_boxes"][0].numpy(),
            "scores": detections["detection_scores"][0].numpy(),
            "classes": detections["detection_classes"][0].numpy().astype(np.int32),
        }
