# Image Preprocessing & Preparation Module

This folder contains the preprocessing pipeline specifications, analytical demonstration figures, and reports for the Fruit and Vegetable Image Recognition dataset.

## Contents
- **[preprocessing_report.md](preprocessing_report.md)**: Full detailed preprocessing methodology report.
- **[figures/](figures/)**: Visual demonstration figures:
  - `letterbox_vs_stretch.png`: Comparison of direct stretching vs letterbox padding.
  - `clahe_illumination_enhancement.png`: Illumination normalization in CIE-LAB color space.
  - `denoise_and_edge_detection.png`: Bilateral filtering & Canny edge extraction.
  - `augmentation_showcase.png`: Demonstration of stochastic data augmentation.
  - `end_to_end_preprocessed_pipeline.png`: Step-by-step pipeline execution flow.

## Code Modules
- **`src/preprocessing.py`**: Core preprocessing classes (`ImagePreprocessor`, `letterbox_resize`, `apply_clahe_lab`, `denoise_image`, `normalize_tensor`).
- **`src/augmentation.py`**: Data augmentation engine (`DataAugmenter`).
- **`src/run_preprocessing_demo.py`**: Script to generate analytical figures.

## Usage
Run demonstration and generate reports:
```bash
python src/run_preprocessing_demo.py
```
