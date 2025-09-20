# CS766-Project
Google doc for proposal: https://docs.google.com/document/d/1NwB6iNb-V5nH1BK8v4i_7VT5IXQuSa25xzvSM1d3G-M/

Website URL: https://weiyuzh.github.io/CS766-Project/

## Cell Detection GUI

The `cell_detection_gui` package provides a desktop application that loads the
TensorFlow object detection pipeline from the project and applies it to
microscopy images. The GUI keeps the model architecture and checkpoint exactly
as described in the project notebooks and documentation under `docs/`.

### Environment setup

1. Create a Python virtual environment (Python 3.10+ is recommended).
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install TensorFlow and the TensorFlow Object Detection API following the
   instructions in the project documentation (`docs/index.md`). The GUI expects
   the ResNet50 FPN checkpoint stored under
   `Machine Learning/Cell Counting Multichannel/checkpoint` to remain in place;
   it will automatically unpack the archived checkpoint files on first use.

### Running the application

```bash
python -m cell_detection_gui
```

### Key features

- Upload one to three sequential microscope slices (PNG, JPG, or TIFF). When a
  single image is supplied the same slice is reused for all three channels.
- Preview the selected slice before running detection.
- Execute the SSD ResNet50 FPN model to obtain cell detections and visualize the
  labeled bounding boxes directly on the original image.
- Review detection metadata (counts, scores, and pixel coordinates) inside the
  application.
- Export annotated images with optional JSON sidecar metadata that captures cell
  locations and confidence scores for downstream analysis.
