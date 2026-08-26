# Image Processing & Recognition Project

Fruit and Vegetable Image Recognition project using Python, OpenCV, and Computer Vision techniques.

## Dataset
- **Dataset Source**: [Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition) (Kaggle)
- **Total Classes**: 36 classes (e.g. apple, banana, beetroot, bell pepper, carrot, mango, tomato, etc.)
- **Splits**:
  - `train`: 36 classes (3,115 images)
  - `validation`: 36 classes (351 images)
  - `test`: 36 classes (359 images)
  - **Total Images**: 3,825 images

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Kaggle API
Set your Kaggle API token or save your token in `~/.kaggle/access_token`:
```bash
# Windows PowerShell
$env:KAGGLE_API_TOKEN = "YOUR_KAGGLE_API_TOKEN"
```

### 3. Execution Pipeline Commands

#### Step 1: Download Dataset
```bash
python src/download_data.py
```

#### Step 2: Run Exploratory Data Analysis (EDA)
```bash
python src/eda.py
```
> Generates comprehensive EDA report and charts in `reports/eda/`.

#### Step 3: Run Preprocessing & Augmentation Demo
```bash
python src/run_preprocessing_demo.py
```
> Generates preprocessing comparative figures in `reports/preprocessing/`.

#### Step 4: Run Data Splitting & Leakage Audit
```bash
python src/data_splitting.py
```
> Audits dataset partitions and generates reports in `reports/data_splitting/`.

## Reports & Documentation
- **EDA Report**: [`reports/eda/eda_report.md`](reports/eda/eda_report.md)
- **Preprocessing Report**: [`reports/preprocessing/preprocessing_report.md`](reports/preprocessing/preprocessing_report.md)
- **Data Splitting Report**: [`reports/data_splitting/data_splitting_report.md`](reports/data_splitting/data_splitting_report.md)

## Project Structure
```text
imageprocessing-prj/
├── data/                               # Dataset directory (ignored in git)
│   ├── train/                          # Training set (36 classes)
│   ├── validation/                     # Validation set (36 classes)
│   └── test/                           # Test set (36 classes)
├── reports/                            # Analytical Reports & Figures
│   ├── eda/                            # Exploratory Data Analysis reports & charts
│   ├── preprocessing/                  # Preprocessing comparison reports & charts
│   └── data_splitting/                 # Data splitting & leakage audit reports
├── src/                                # Source code directory
│   ├── __init__.py
│   ├── download_data.py                # Dataset downloader via Kaggle API
│   ├── eda.py                          # Exploratory Data Analysis pipeline
│   ├── preprocessing.py                # Letterboxing, CLAHE, Denoising, Normalization
│   ├── augmentation.py                 # Geometric & Photometric Data Augmentation
│   ├── run_preprocessing_demo.py       # Preprocessing demonstration visualizer
│   └── data_splitting.py               # Splitting audit & clean stratified splitter
├── requirements.txt                    # Project Python dependencies
├── .gitignore                          # Git ignore configuration
└── README.md                           # Project documentation
```
