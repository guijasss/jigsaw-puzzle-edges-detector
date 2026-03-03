# Jigsaw Puzzle Edge Detector

Simple Python + OpenCV GUI app to detect and highlight edge pieces in a jigsaw puzzle image. Includes basic instance segmentation (watershed) to separate partially overlapping pieces.

## Requirements

- Python 3.10+
- OpenCV
- NumPy
- Pillow

## Setup

```bash
python -m venv .venv
```

```bash
.venv\\Scripts\\activate
```

```bash
pip install opencv-python numpy pillow
```

## Run

```bash
python src\\main.py
```

## Notes

- Upload an image and click "Detect Edge Pieces" to see highlights.
- Overlapping pieces are separated with a watershed pass before classification.
