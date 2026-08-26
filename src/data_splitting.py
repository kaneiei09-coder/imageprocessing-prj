"""
Data Splitting and Validation Module for Fruit and Vegetable Image Recognition.
Audits existing dataset splits, checks class stratification across all 36 classes,
detects potential data leakage via image hashing, and provides flexible re-splitting utilities.
"""

import sys
import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split

# Ensure UTF-8 output for Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports" / "data_splitting"
FIGURES_DIR = REPORT_DIR / "figures"

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Arial"

def compute_file_hash(file_path: Path) -> str:
    """Computes MD5 hash of a file for duplicate detection."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def audit_dataset_splits(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, dict]:
    """
    Audits the existing train, validation, and test partitions in data_dir.
    Returns (records_dataframe, audit_summary_dictionary).
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    splits = ["train", "validation", "test"]
    records = []
    hash_map: Dict[str, List[dict]] = {}

    print(f"Auditing dataset partitions in '{data_dir}'...")

    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"Warning: Partition '{split}' does not exist.")
            continue

        classes = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        for cat in classes:
            class_name = cat.name
            files = [f for f in cat.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]

            for img_f in files:
                file_hash = compute_file_hash(img_f)
                rec = {
                    "split": split,
                    "class": class_name,
                    "filename": img_f.name,
                    "filepath": str(img_f),
                    "file_size_kb": round(img_f.stat().st_size / 1024.0, 2),
                    "hash": file_hash
                }
                records.append(rec)

                if file_hash not in hash_map:
                    hash_map[file_hash] = []
                hash_map[file_hash].append(rec)

    df = pd.DataFrame(records)

    # Detect Cross-Split Leakage (identical files in multiple splits)
    leakage_records = []
    for f_hash, entries in hash_map.items():
        if len(entries) > 1:
            splits_present = set(e["split"] for e in entries)
            if len(splits_present) > 1:
                leakage_records.append({
                    "hash": f_hash,
                    "occurrences": len(entries),
                    "splits": list(splits_present),
                    "files": [e["filepath"] for e in entries]
                })

    total_samples = len(df)
    unique_samples = df["hash"].nunique()
    split_counts = df["split"].value_counts().to_dict()
    classes_per_split = {s: df[df["split"] == s]["class"].nunique() for s in df["split"].unique()}

    audit_summary = {
        "total_samples": total_samples,
        "unique_samples": unique_samples,
        "duplicate_instances": total_samples - unique_samples,
        "split_counts": split_counts,
        "split_ratios": {s: round((c / total_samples) * 100, 2) for s, c in split_counts.items()},
        "total_classes": df["class"].nunique(),
        "classes_per_split": classes_per_split,
        "cross_split_leakage_count": len(leakage_records),
        "leakage_details": leakage_records
    }

    return df, audit_summary

def generate_split_visualizations(df: pd.DataFrame, audit_summary: dict, output_dir: Path = FIGURES_DIR):
    """
    Generates high-resolution visualization figures for the data splitting report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating data splitting figures in '{output_dir}'...")

    colors = {"train": "#2b5c8f", "validation": "#e28743", "test": "#479b73"}

    # 1. Stratified Distribution Bar Chart (Counts per class per split)
    plt.figure(figsize=(13, 12))
    pivot = df.groupby(["class", "split"]).size().unstack(fill_value=0)
    cols = [c for c in ["train", "validation", "test"] if c in pivot.columns]
    pivot = pivot[cols]

    pivot.plot(
        kind="barh",
        stacked=True,
        figsize=(12, 12),
        color=[colors.get(c, "#888888") for c in cols],
        width=0.75
    )
    plt.title("Class Stratification Across Partitions (Train / Validation / Test)", fontsize=14, weight="bold", pad=15)
    plt.xlabel("Sample Count", fontsize=12)
    plt.ylabel("Category (Class)", fontsize=12)
    plt.legend(title="Split Partition", frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "stratified_split_distribution.png", dpi=300)
    plt.close()

    # 2. Split Ratio Donut Chart
    plt.figure(figsize=(7, 7))
    split_counts = df["split"].value_counts()
    split_colors = [colors.get(s, "#555555") for s in split_counts.index]
    plt.pie(
        split_counts,
        labels=[f"{s.capitalize()}\n{c:,} imgs ({audit_summary['split_ratios'][s]:.1f}%)" for s, c in zip(split_counts.index, split_counts)],
        autopct="%1.1f%%",
        startangle=140,
        colors=split_colors,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        textprops={"fontsize": 11, "weight": "bold"}
    )
    plt.title(f"Dataset Split Ratio Architecture (Total: {len(df):,} Images)", fontsize=13, weight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "split_ratio_summary.png", dpi=300)
    plt.close()

    # 3. Class Stratification Heatmap Matrix
    plt.figure(figsize=(10, 14))
    heatmap_data = pivot[cols]
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="Blues", cbar=True, linewidths=0.5)
    plt.title("Per-Class Sample Distribution Matrix across Splits", fontsize=13, weight="bold", pad=15)
    plt.xlabel("Split Partition", fontsize=11, weight="bold")
    plt.ylabel("Class Name", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "class_stratification_heatmap.png", dpi=300)
    plt.close()

    # 4. Data Leakage vs Unique Samples Comparison Bar
    plt.figure(figsize=(8, 5))
    metrics = ["Total Raw Images", "Unique Images", "Duplicate Collisions"]
    vals = [audit_summary["total_samples"], audit_summary["unique_samples"], audit_summary["duplicate_instances"]]
    bar_colors = ["#2b5c8f", "#479b73", "#d9534f"]
    
    bars = plt.bar(metrics, vals, color=bar_colors, width=0.55)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{yval:,}", ha="center", va="bottom", weight="bold", fontsize=11)
        
    plt.title("Data Integrity & Deduplication Audit", fontsize=13, weight="bold", pad=15)
    plt.ylabel("Image Count", fontsize=11)
    plt.ylim(0, max(vals) * 1.15)
    plt.tight_layout()
    plt.savefig(output_dir / "data_leakage_audit.png", dpi=300)
    plt.close()

    print("All data splitting figures created successfully.")

def create_clean_stratified_split(
    df: pd.DataFrame, 
    dest_dir: Path,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_state: int = 42
):
    """
    Creates a clean, deduplicated, stratified train/validation/test split from unique images.
    """
    print(f"\nCreating clean stratified split in '{dest_dir}'...")
    # Deduplicate by image hash
    unique_df = df.drop_duplicates(subset=["hash"]).reset_index(drop=True)
    
    # Train / Temp Split
    train_df, temp_df = train_test_split(
        unique_df, 
        test_size=(val_ratio + test_ratio), 
        stratify=unique_df["class"], 
        random_state=random_state
    )
    
    # Val / Test Split
    rel_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=rel_test_ratio, 
        stratify=temp_df["class"], 
        random_state=random_state
    )
    
    splits_dict = {"train": train_df, "validation": val_df, "test": test_df}
    
    print(f"Clean Stratified Split created:")
    for s_name, s_df in splits_dict.items():
        print(f"  [{s_name.upper():<10}] - {len(s_df)} unique images ({len(s_df)/len(unique_df)*100:.1f}%) across {s_df['class'].nunique()} classes")

    return splits_dict

def run_data_splitting_audit():
    """
    Main function to run the data splitting audit and generate analytical reports.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df, summary = audit_dataset_splits(DATA_DIR)

    # Save split audit CSV
    df.to_csv(REPORT_DIR / "split_audit_records.csv", index=False)

    # Generate Visual Figures
    generate_split_visualizations(df, summary, FIGURES_DIR)

    # Compute clean stratified split statistics
    clean_splits = create_clean_stratified_split(df, DATA_DIR / "clean_split_preview")

    print("\n" + "=" * 60)
    print("DATA SPLITTING AUDIT COMPLETE")
    print("=" * 60)
    print(f"Total Dataset Images:     {summary['total_samples']}")
    print(f"Unique Dataset Images:    {summary['unique_samples']}")
    print(f"Cross-Split Hash Leaks:   {summary['cross_split_leakage_count']} duplicate hash collisions")
    print(f"Split Proportions:        {summary['split_ratios']}")
    print(f"Split Sample Counts:      {summary['split_counts']}")
    print(f"Classes per Split:        {summary['classes_per_split']}")
    print("=" * 60 + "\n")

    return df, summary

if __name__ == "__main__":
    run_data_splitting_audit()
