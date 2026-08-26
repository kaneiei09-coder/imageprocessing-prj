# Dataset Partitioning & Splitting Validation Module

This folder contains the audit reports, class stratification matrices, duplicate hash detection analysis, and figures for the Fruit and Vegetable Image Recognition dataset.

## Contents
- **[data_splitting_report.md](data_splitting_report.md)**: Comprehensive validation report.
- **[split_audit_records.csv](split_audit_records.csv)**: Complete audit records with MD5 hashes.
- **[figures/](figures/)**:
  - `stratified_split_distribution.png`: Per-class sample counts across Train, Validation, and Test partitions.
  - `split_ratio_summary.png`: Proportion chart of dataset splits.
  - `class_stratification_heatmap.png`: Matrix heatmap of sample counts per class.
  - `data_leakage_audit.png`: Duplicate hash collision and data leakage analysis chart.

## Code Modules
- **`src/data_splitting.py`**: Dataset partition auditor, hash duplicate scanner, and clean stratified re-splitting engine.

## Usage
Run data splitting audit:
```bash
python src/data_splitting.py
```
