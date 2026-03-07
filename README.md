# X-Ray Security Screening Challenge


Detect prohibited items in X-ray baggage scans using **classical computer vision and machine learning only** (no deep learning). Built for the Kaggle [X-Ray Security Screening Challenge](https://www.kaggle.com/competitions/vic-vision-par-ordinateur-object-detection-in-baggage-using-classic) that serves as the Assignment 2 of the "Introduction to Visual Computing" course at CentraleSupélec.

The system identifies 6 classes of dangerous objects: **Hammer, Knife, Gun, Wrench, HandCuffs, Bullet** from X-ray images of luggage, using a two-stage detection pipeline inspired by R-CNN but powered entirely by classical techniques.

## Overview

- **No deep learning constraint**: demonstrates what classical CV can achieve on a real object detection task
- **Two-stage architecture**: ensemble region proposals + Random Forest classification with per-class threshold tuning
- **Rich feature engineering**: HOG descriptors, Local Binary Patterns, intensity histograms, Hu Moments, Canny edge maps, and Hough line features
- **End-to-end pipeline**: from data loading and exploration to model training, evaluation, and Kaggle submission

## Project Structure

```
├── src/xrss/               # Core library
│   ├── dataloader.py        # XRayDataset loader (YOLO-format labels) + visualization
│   ├── main_model.py        # TwoStageDetector (final model)
│   ├── old_models.py        # PixelTemplateMatching (baseline)
│   └── utils.py             # BBox conversions, IoU, evaluation scoring
├── notebooks/
│   ├── Visualisation.ipynb          # Dataset exploration and bounding box visualization
│   ├── PixelTemplateMatching.ipynb  # Baseline: brute-force template matching
│   ├── ImprovedPTM.ipynb            # Multi-scale/rotation template matching
│   ├── MetalMaskRandomForest.ipynb  # HSV metal segmentation + Random Forest
│   └── TwoStageDetector.ipynb       # Final model: two-stage ensemble detector
├── predictions/             # Per-image prediction outputs (YOLO format)
├── results/                 # Kaggle submission CSVs
├── report/                  # LaTeX project report
├── xray_data/               # Dataset (images + YOLO labels)
│   ├── data.yaml            # Dataset config (classes, paths, splits)
│   ├── images/{train,val,test}/
│   └── labels/{train,val}/
├── pyproject.toml
└── requirements.txt
```

## Getting Started

### Requirements

- Python ≥ 3.10

### Installation

```bash
git clone https://github.com/JeanJNMV/MDS-VIC_XRay_Challenge.git
cd MDS-VIC_XRay_Challenge

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Alternatively, the dependencies can be installed via **uv** by running `uv sync` in your terminal.

### Quick Example

```python
from xrss.dataloader import XRayDataset, show_images_and_bboxes
from xrss.main_model import TwoStageDetector
import numpy as np

# Load data
dataset = XRayDataset("xray_data/data.yaml", split="train")
val_dataset = XRayDataset("xray_data/data.yaml", split="val")

# Train the two-stage detector
model = TwoStageDetector(nc=6)
model.train(dataset)

# Run detection on a single image
img, gt_labels = val_dataset[0]
predictions = model.detect(np.array(img))

# Visualize results
show_images_and_bboxes(val_dataset, [img], [predictions])
```

### Evaluation

```python
from xrss.utils import evaluate_score

score = evaluate_score(model, val_dataset)
print(f"Mean IoU score: {score:.4f}")
```

## Notebooks

| Notebook | Approach | Description |
|----------|----------|-------------|
| `Visualisation` | — | Dataset exploration with bounding box overlays |
| `PixelTemplateMatching` | Baseline | Normalized cross-correlation with ~700 templates/class; NMS grid search |
| `ImprovedPTM` | Enhanced baseline | Multi-scale (0.6–1.4×), multi-rotation template matching |
| `MetalMaskRandomForest` | HSV segmentation | Metal region extraction via HSV thresholding + geometric feature classifier |
| `TwoStageDetector` | **Final model** | Ensemble region proposals (5 methods) → binary RF → multi-class RF with per-class thresholds |

The recommended starting point is the `TwoStageDetector` notebook, which contains the full training and evaluation pipeline. The earlier notebooks show the iterative development process, starting from a simple template matching baseline and progressively adding complexity through improved techniques and feature engineering, culminating in the final two-stage detection system.

## How It Works

The final **Two-Stage Detector** works as follows:

1. **Region Proposals** — 5 complementary segmentation methods (multi-thresholding, percentile thresholding, CLAHE, adaptive thresholding, MSER) generate candidate regions, merged via NMS
2. **Stage 1 — Binary Classification** — A Random Forest filters proposals into *object* vs *background* using HOG, LBP, edge, and shape features
3. **Stage 2 — Multi-class Classification** — A second Random Forest classifies surviving proposals into one of 6 object classes, with per-class confidence thresholds and class weights to handle imbalance (e.g., bullets are rare and small)

## Author

Jean-Vincent Martini — CentraleSupélec

