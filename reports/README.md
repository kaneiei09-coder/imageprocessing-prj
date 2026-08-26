# Project Reports Directory

This folder contains all analytical reports, data audits, and visual figures produced by the Fruit and Vegetable Image Processing pipeline.

## Master Project Report
- **[PROJECT_REPORT.md](PROJECT_REPORT.md)**: Comprehensive end-to-end master report covering the entire project lifecycle.

## Specialized Reports
1. **[Exploratory Data Analysis (EDA)](eda/eda_report.md)**:
   - Statistical properties, resolution spreads, aspect ratios, color channel distributions.
   - [dataset_metadata.csv](eda/dataset_metadata.csv): Per-image metadata.
   - [figures/](eda/figures/): 6 visualization figures.

2. **[Image Preprocessing & Augmentation](preprocessing/preprocessing_report.md)**:
   - Letterboxing vs direct stretch, CIE-LAB CLAHE contrast enhancement, Bilateral denoising, Z-score normalization.
   - [figures/](preprocessing/figures/): 5 comparative visualization figures.

3. **[Data Splitting & Leakage Audit](data_splitting/data_splitting_report.md)**:
   - Partition audit (Train/Val/Test), class stratification matrix, cryptographic MD5 duplicate detector.
   - [split_audit_records.csv](data_splitting/split_audit_records.csv): Per-image hash records.
   - [figures/](data_splitting/figures/): 4 visualization figures.
