# Exploratory Data Analysis (EDA) Module & Reports

This directory contains the analytical findings, statistical distributions, visual figures, and metadata generated from the Exploratory Data Analysis of the Fruit and Vegetable Image Recognition dataset.

## Contents
- **[eda_report.md](eda_report.md)**: Full detailed EDA report in Markdown format.
- **[dataset_metadata.csv](dataset_metadata.csv)**: Complete image-level metadata (resolution, aspect ratio, file size, RGB/HSV channels, split, class).
- **[figures/](figures/)**: Visual charts and plots:
  - `class_distribution.png` - Distribution across all 36 classes and splits.
  - `split_proportions.png` - Proportion of Train, Validation, and Test subsets.
  - `dimension_distribution.png` - Width, height, and aspect ratio histograms.
  - `dimension_scatter.png` - Width vs Height resolution scatter plot.
  - `color_channel_distribution.png` - RGB and HSV color intensity distributions.
  - `sample_classes_grid.png` - Sample grid of 16 fruit/vegetable classes.

## Execution
To reproduce the analysis and re-generate reports/figures:
```bash
python src/eda.py
```
