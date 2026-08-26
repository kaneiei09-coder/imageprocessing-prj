# Fruit and Vegetable Image Recognition & Processing

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Computer Vision and Image Processing pipeline for multi-class Fruit and Vegetable classification using Python and OpenCV. Includes automated data ingestion, extensive Exploratory Data Analysis (EDA), morphology-preserving preprocessing (letterboxing, CIE-LAB CLAHE, bilateral denoising), cryptographic data leakage auditing, and one-click batch automation.

---

## ⚡ Quick Start: One-Click Pipeline Execution

Run the complete pipeline (dependencies, data retrieval, EDA, preprocessing demo, and splitting audit) with a single command on Windows:

```cmd
run_pipeline.bat
```

*(or run the individual Python modules listed below)*.

---

## 📊 Dataset Overview
- **Source**: [Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition) (Kaggle)
- **Total Sample Size**: **3,825 images**
- **Unique Produce Categories**: **36 classes** (e.g., apple, banana, beetroot, bell pepper, carrot, mango, tomato, watermelon, etc.)
- **Partitioning**:
  - `train`: 3,115 images (81.44%)
  - `validation`: 351 images (9.18%)
  - `test`: 359 images (9.38%)

---

## 🚀 Step-by-Step Execution Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Kaggle API (Optional if already configured)
Set your Kaggle API token or store it in `~/.kaggle/access_token`:
```powershell
$env:KAGGLE_API_TOKEN = "YOUR_KAGGLE_API_TOKEN"
```

### 3. Pipeline Stages

#### Stage 1: Download & Ingest Dataset
```bash
python src/download_data.py
```
> Downloads the dataset via `kagglehub` and organizes it locally into `data/train`, `data/validation`, and `data/test`.

#### Stage 2: Exploratory Data Analysis (EDA)
```bash
python src/eda.py
```
> Analyzes class distributions, spatial resolutions ($133$ px to $7,360$ px), aspect ratios ($0.40$ to $3.45$), RGB/HSV color channels, and saves 6 high-res analytical figures in `reports/eda/figures/`.

#### Stage 3: Image Preprocessing & Augmentation Demo
```bash
python src/run_preprocessing_demo.py
```
> Demonstrates aspect-ratio preserving letterboxing (224x224), CIE-LAB CLAHE illumination normalization, bilateral filtering, Canny edge detection, and stochastic augmentation. Saves 5 comparison figures in `reports/preprocessing/figures/`.

#### Stage 4: Data Splitting & Leakage Audit
```bash
python src/data_splitting.py
```
> Audits class stratification uniformity and executes cryptographic MD5 hash scans to detect data leakage. Saves 4 analytical figures in `reports/data_splitting/figures/`.

---

## 📑 Analytical Reports & Documentation

| Document | Description |
| :--- | :--- |
| **[PROJECT_REPORT.md](reports/PROJECT_REPORT.md)** | **Master Project Report**: Executive summary covering the complete lifecycle, architecture, and findings. |
| **[eda_report.md](reports/eda/eda_report.md)** | **EDA Report**: Deep-dive into class distributions, aspect ratio variations, and color space dynamics. |
| **[preprocessing_report.md](reports/preprocessing/preprocessing_report.md)** | **Preprocessing Report**: Technical rationale for letterboxing, CLAHE in LAB space, and augmentation. |
| **[data_splitting_report.md](reports/data_splitting/data_splitting_report.md)** | **Data Splitting Report**: Audit of existing partitions, duplicate hash leakage discovery, and clean stratified split. |

---

## 📁 Project Directory Structure

```text
imageprocessing-prj/
├── data/                               # Dataset directory (ignored in git)
│   ├── train/                          # 36 classes (3,115 images)
│   ├── validation/                     # 36 classes (351 images)
│   └── test/                           # 36 classes (359 images)
├── reports/                            # Analytical Reports & Figures
│   ├── README.md                       # Reports index
│   ├── PROJECT_REPORT.md               # Master comprehensive project report
│   ├── eda/                            # EDA reports, metadata CSV, and 6 figures
│   ├── preprocessing/                  # Preprocessing reports and 5 figures
│   └── data_splitting/                 # Splitting & leakage audit reports and 4 figures
├── src/                                # Source code directory
│   ├── __init__.py
│   ├── download_data.py                # Dataset downloader via Kaggle API
│   ├── eda.py                          # Exploratory Data Analysis pipeline
│   ├── preprocessing.py                # Letterboxing, CLAHE, Denoising, Normalization
│   ├── augmentation.py                 # Geometric & Photometric Data Augmentation
│   ├── run_preprocessing_demo.py       # Preprocessing demonstration visualizer
│   └── data_splitting.py               # Splitting audit & clean stratified splitter
├── run_pipeline.bat                    # One-click Windows batch pipeline runner
├── requirements.txt                    # Python package dependencies
├── .gitignore                          # Git ignore rules (dataset, cache, env)
└── README.md                           # Master repository documentation
```

---

## 🌿 Git Branching History

This project was built following a professional feature-branch workflow:
1. `feat/datacollection`: Dataset retrieval and project reorganization into `src/`.
2. `feat/exploratorydataanalysis`: Comprehensive statistical & visual EDA.
3. `feat/imagepreprocessing`: Modular preprocessing and augmentation pipeline.
4. `feat/datasplitting`: Stratification audit and cryptographic leakage analysis.
5. `main`: Master integration, final reports, and `run_pipeline.bat` automation.
