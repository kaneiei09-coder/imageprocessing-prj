# Image Processing & Recognition Project

Fruit and Vegetable Image Recognition project using Python, OpenCV, and Image Processing techniques.

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

### 3. Download Dataset
Run the data collection script to fetch and extract images directly into the local `data/` folder:
```bash
python download_data.py
```

## Project Structure
```text
imageprocessing-prj/
├── data/                    # Dataset directory (ignored in git)
│   ├── train/               # Training set (36 classes)
│   ├── validation/          # Validation set (36 classes)
│   └── test/                # Test set (36 classes)
├── download_data.py         # Script to download dataset from Kaggle
├── requirements.txt         # Project Python dependencies
├── .gitignore               # Git ignore configuration
└── README.md                # Project documentation
```
