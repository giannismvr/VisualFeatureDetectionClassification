# Visual Feature Detection and Classification

A complete computer-vision project implemented in a Jupyter notebook, covering:

1. **Edge detection under noise**
2. **Corner/blob detection with multiscale selection**
3. **Image classification with local features + Bag of Visual Words + SVM**

The main deliverable is the notebook `VisualFeatureDetectionClassification.ipynb`, supported by reusable utility scripts and dataset folders.

---

## Table of Contents

- [Project Goals](#project-goals)
- [What This Repository Contains](#what-this-repository-contains)
- [Methodology Overview](#methodology-overview)
  - [Part 1 - Edge Detection and Evaluation](#part-1---edge-detection-and-evaluation)
  - [Part 2 - Interest Point Detection](#part-2---interest-point-detection)
  - [Part 3 - Visual Classification Pipeline](#part-3---visual-classification-pipeline)
- [Repository Structure](#repository-structure)
- [Environment and Dependencies](#environment-and-dependencies)
- [How to Run](#how-to-run)
  - [Option A: Run the Notebook](#option-a-run-the-notebook)
  - [Option B: Run Scripted Classification Example](#option-b-run-scripted-classification-example)
- [Experimental Design Notes](#experimental-design-notes)
- [Expected Outputs](#expected-outputs)
- [Reproducibility](#reproducibility)
- [Known Limitations](#known-limitations)

---

## Project Goals

This project demonstrates an end-to-end feature-engineering workflow in classical computer vision:

- Build and compare edge detectors in noisy conditions.
- Detect corners and blobs at multiple scales.
- Use detected keypoints for image representation and category recognition.
- Evaluate a multiclass classifier on the TUGraz object categories (bike, cars, person).

The implementation balances **algorithmic clarity** and **practical experimentation** for coursework/lab settings.

---

## What This Repository Contains

- A single notebook (`VisualFeatureDetectionClassification.ipynb`) that integrates all parts.
- Utility modules for plotting, detector support, descriptor extraction, train/test splitting, BoW encoding, and SVM classification.
- Data/material folders with sample images and fold indices used by the lab pipeline.

---

## Methodology Overview

## Part 1 - Edge Detection and Evaluation

Main ideas implemented in the notebook:

- Gaussian noise injection at controlled levels.
- Gaussian smoothing with a custom 2D kernel (`Gauss2D`).
- Two LoG-style edge pipelines:
  - **Linear** LoG response.
  - **Nonlinear** response based on morphological operations.
- Zero-crossing edge extraction with thresholding.
- Quantitative scoring via a custom evaluation function (`Evaluation`) combining precision/recall style criteria.
- Parameter sweeps over `sigma` and `theta_edge` to find best operating points for different noise levels.

This section emphasizes robustness and parameter sensitivity under degraded imaging conditions.

## Part 2 - Interest Point Detection

Implemented detectors and scale-space logic:

- Harris-style corner response from the structure tensor.
- Hessian determinant for blob response.
- Non-maximum suppression + thresholding in image space.
- Multiscale selection using normalized LoG responses (`sigma^2 |LoG|`).
- Unified keypoint format `(x, y, scale)` for downstream descriptors.

Core notebook functions include:

- `CornerDetect`
- `BlobDetect`
- `MultiscaleCornerDetect`
- `MultiscaleBlobDetect`
- `harrisLaplaceDetector`
- `hessianLaplaceDetector`

## Part 3 - Visual Classification Pipeline

The final stage turns local features into class predictions:

1. Detect keypoints (Harris-Laplace or Hessian-Laplace variants).
2. Describe keypoints with either:
   - `featuresSURF` (implemented using OpenCV SIFT descriptor extraction on supplied keypoints)
   - `featuresHOG` (custom local patch HOG)
3. Build fixed-length image vectors using **Bag of Visual Words** (`BagOfWords`).
4. Train/evaluate multiclass **One-vs-Rest linear SVM** (`svm`).
5. Run over predefined folds from `Fold_Indices.mat` for reproducible evaluation.

The notebook experiments with detector/descriptor combinations and reports mean performance across folds.

---

## Repository Structure

```text
VisualFeatureDetectionClassification/
|-- VisualFeatureDetectionClassification.ipynb   # Main end-to-end notebook
|-- material/                                    # Utility scripts + lab material
|   |-- cv26_lab1_part2_utils.py
|   |-- cv26_lab1_part3_utils.py
|   |-- Part_2.py
|   |-- example_classification.py
|   `-- Fold_Indices.mat
|-- cv26_lab1_part12_material/                   # Part 1/2 support data and utils
|-- cv26_lab1_part3_material/                    # Part 3 support scripts/material
`-- Data/                                        # Dataset and sample images (ignored by default)
```

> Note: `Data/` is typically excluded from version control via `.gitignore`.

---

## Environment and Dependencies

Recommended: Python 3.9+.

Main packages used by the notebook/scripts:

- `numpy`
- `scipy`
- `matplotlib`
- `opencv-contrib-python` (OpenCV + SIFT/SURF-compatible APIs)
- `scikit-learn`
- `tqdm`
- `Pillow`
- `torch`, `torchvision` (used in parts of the notebook)
- `jupyter`

Suggested installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib scikit-learn opencv-contrib-python tqdm pillow torch torchvision jupyter
```

---

## How to Run

## Option A: Run the Notebook

```bash
source .venv/bin/activate
jupyter notebook VisualFeatureDetectionClassification.ipynb
```

Run cells sequentially to preserve variable state and dependencies between parts.

## Option B: Run Scripted Classification Example

```bash
source .venv/bin/activate
python material/example_classification.py
```

If paths differ in your local setup, adjust imports/relative paths accordingly.

---

## Experimental Design Notes

- **Fold-based splitting** is taken from `Fold_Indices.mat` for consistency across runs.
- **BoW vocabulary size** and clustering sample ratio directly affect runtime/accuracy trade-offs.
- **Detector thresholds and scale factors** significantly influence keypoint quality and downstream classification.
- **Image downsampling** in feature extraction improves runtime and is intentional.

---

## Expected Outputs

From the notebook and scripts, you should expect:

- Visualization figures for edge maps, corners, blobs, and multiscale detections.
- Parameter search outcomes for noise conditions (best `sigma` and threshold values).
- Per-fold classification results and aggregate mean accuracy for detector/descriptor combinations.

---

## Reproducibility

To reproduce results consistently:

- Keep the same dataset organization under `Data/`.
- Use the provided fold indices (`Fold_Indices.mat`).
- Run with the same package versions (or pin them in a `requirements.txt`).
- Avoid reordering notebook cells during execution.

---

## Known Limitations

- Runtime can become high for full-feature extraction and clustering.
- Performance depends on threshold/scale choices and BoW hyperparameters.
- Some OpenCV descriptor APIs differ by version/build; `opencv-contrib-python` is recommended.
- Notebook-style workflows can hide state issues if cells are run out of order.

---


