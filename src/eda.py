"""
Exploratory Data Analysis (EDA) for Fruit and Vegetable Image Recognition Dataset.
Analyzes dataset architecture, class distributions, image dimensions, aspect ratios,
color channel distributions, and generates visual analytical figures.
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Ensure UTF-8 output for Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports" / "eda"
FIGURES_DIR = REPORT_DIR / "figures"

# Matplotlib styling for publication-quality figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

def read_image_rgb(file_path: str | Path) -> np.ndarray | None:
    """
    Robustly reads an image file supporting Unicode filepaths on Windows.
    Returns RGB numpy array or None if corrupt.
    """
    try:
        with Image.open(file_path) as pil_img:
            rgb_img = pil_img.convert("RGB")
            return np.array(rgb_img)
    except Exception:
        return None

def collect_image_metadata(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Scans the dataset directory and extracts comprehensive metadata for each image.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    records = []
    
    splits = ["train", "validation", "test"]
    print("Collecting metadata across all dataset splits...")
    
    for split in splits:
        split_path = data_dir / split
        if not split_path.exists():
            print(f"Warning: Split path '{split_path}' does not exist.")
            continue
            
        categories = sorted([d for d in split_path.iterdir() if d.is_dir()])
        for cat in categories:
            class_name = cat.name
            image_files = [f for f in cat.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
            
            for img_file in image_files:
                file_size_kb = img_file.stat().st_size / 1024.0
                rgb_arr = read_image_rgb(img_file)
                
                if rgb_arr is not None:
                    h, w, c = rgb_arr.shape
                    is_corrupt = False
                    
                    # RGB channel statistics
                    r_mean, g_mean, b_mean = rgb_arr.mean(axis=(0, 1))
                    r_std, g_std, b_std = rgb_arr.std(axis=(0, 1))
                    
                    # HSV statistics
                    hsv_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)
                    h_mean, s_mean, v_mean = hsv_arr.mean(axis=(0, 1))
                    
                    # File format
                    with Image.open(img_file) as pil_f:
                        img_format = pil_f.format or img_file.suffix.upper().replace(".", "")
                else:
                    w = h = c = 0
                    img_format = "CORRUPT"
                    r_mean = g_mean = b_mean = r_std = g_std = b_std = 0
                    h_mean = s_mean = v_mean = 0
                    is_corrupt = True

                records.append({
                    "split": split,
                    "class": class_name,
                    "filename": img_file.name,
                    "filepath": str(img_file),
                    "file_size_kb": file_size_kb,
                    "width": w,
                    "height": h,
                    "aspect_ratio": round(w / h, 4) if h > 0 else 0,
                    "channels": c,
                    "format": img_format,
                    "r_mean": round(r_mean, 2),
                    "g_mean": round(g_mean, 2),
                    "b_mean": round(b_mean, 2),
                    "h_mean": round(h_mean, 2),
                    "s_mean": round(s_mean, 2),
                    "v_mean": round(v_mean, 2),
                    "is_corrupt": is_corrupt
                })

    df = pd.DataFrame(records)
    print(f"Metadata collected for {len(df)} images.")
    return df

def generate_visualizations(df: pd.DataFrame, output_dir: Path = FIGURES_DIR):
    """
    Generates and saves all EDA analytical charts and visualizations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating EDA visual figures in '{output_dir}'...")

    colors = {"train": "#2b5c8f", "validation": "#e28743", "test": "#479b73"}

    # 1. Class Distribution Bar Chart (Train / Val / Test)
    plt.figure(figsize=(12, 12))
    class_order = sorted(df["class"].unique())
    pivot_df = df.groupby(["class", "split"]).size().unstack(fill_value=0)
    pivot_df = pivot_df.reindex(class_order)
    
    cols = [c for c in ["train", "validation", "test"] if c in pivot_df.columns]
    pivot_df = pivot_df[cols]
    
    pivot_df.plot(
        kind="barh", 
        stacked=True, 
        figsize=(12, 12), 
        color=[colors.get(c, "#888888") for c in cols],
        width=0.75
    )
    plt.title("Class Distribution Across Splits (Train / Validation / Test)", fontsize=14, weight="bold", pad=15)
    plt.xlabel("Number of Images", fontsize=12)
    plt.ylabel("Fruit / Vegetable Class", fontsize=12)
    plt.legend(title="Split", frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=300)
    plt.close()

    # 2. Overall Split Proportions (Pie / Donut Chart)
    plt.figure(figsize=(7, 7))
    split_counts = df["split"].value_counts()
    split_colors = [colors.get(s, "#555555") for s in split_counts.index]
    plt.pie(
        split_counts, 
        labels=[f"{s.capitalize()} ({c:,} imgs)" for s, c in zip(split_counts.index, split_counts)],
        autopct="%1.1f%%",
        startangle=140,
        colors=split_colors,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        textprops={"fontsize": 11, "weight": "bold"}
    )
    plt.title(f"Dataset Split Proportions (Total: {len(df):,} Images)", fontsize=13, weight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "split_proportions.png", dpi=300)
    plt.close()

    # 3. Image Dimensions Distribution (Histograms & Boxplots)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    sns.histplot(df["width"], kde=True, ax=axes[0], color="#2b5c8f", bins=30)
    axes[0].set_title("Image Width Distribution", fontsize=12, weight="bold")
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(df["width"].median(), color="red", linestyle="--", label=f"Median: {df['width'].median():.0f}px")
    axes[0].legend()

    sns.histplot(df["height"], kde=True, ax=axes[1], color="#e28743", bins=30)
    axes[1].set_title("Image Height Distribution", fontsize=12, weight="bold")
    axes[1].set_xlabel("Height (pixels)")
    axes[1].set_ylabel("Count")
    axes[1].axvline(df["height"].median(), color="red", linestyle="--", label=f"Median: {df['height'].median():.0f}px")
    axes[1].legend()

    sns.histplot(df["aspect_ratio"], kde=True, ax=axes[2], color="#479b73", bins=30)
    axes[2].set_title("Aspect Ratio (Width / Height)", fontsize=12, weight="bold")
    axes[2].set_xlabel("Aspect Ratio")
    axes[2].set_ylabel("Count")
    axes[2].axvline(1.0, color="darkred", linestyle="--", label="1:1 (Square)")
    axes[2].axvline(df["aspect_ratio"].median(), color="blue", linestyle=":", label=f"Median: {df['aspect_ratio'].median():.2f}")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "dimension_distribution.png", dpi=300)
    plt.close()

    # 4. Width vs Height Scatter with Aspect Ratio Contours
    plt.figure(figsize=(9, 8))
    sns.scatterplot(
        data=df, 
        x="width", 
        y="height", 
        hue="split", 
        palette=colors, 
        alpha=0.6, 
        s=35
    )
    max_dim = max(df["width"].max(), df["height"].max())
    plt.plot([0, max_dim], [0, max_dim], "k--", alpha=0.5, label="1:1 Aspect Ratio (Square)")
    plt.title("Image Resolution Distribution (Width vs. Height)", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Width (px)", fontsize=11)
    plt.ylabel("Height (px)", fontsize=11)
    plt.legend(title="Split", frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "dimension_scatter.png", dpi=300)
    plt.close()

    # 5. Color Channel Intensity Distribution (RGB and HSV)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    
    # RGB Means
    sns.histplot(df["r_mean"], ax=axes[0, 0], color="red", kde=True, bins=30)
    axes[0, 0].set_title(f"Red Channel Mean (Avg: {df['r_mean'].mean():.1f})", fontsize=11, weight="bold")
    axes[0, 0].set_xlabel("Pixel Intensity [0-255]")

    sns.histplot(df["g_mean"], ax=axes[0, 1], color="green", kde=True, bins=30)
    axes[0, 1].set_title(f"Green Channel Mean (Avg: {df['g_mean'].mean():.1f})", fontsize=11, weight="bold")
    axes[0, 1].set_xlabel("Pixel Intensity [0-255]")

    sns.histplot(df["b_mean"], ax=axes[0, 2], color="blue", kde=True, bins=30)
    axes[0, 2].set_title(f"Blue Channel Mean (Avg: {df['b_mean'].mean():.1f})", fontsize=11, weight="bold")
    axes[0, 2].set_xlabel("Pixel Intensity [0-255]")

    # HSV Means
    sns.histplot(df["h_mean"], ax=axes[1, 0], color="purple", kde=True, bins=30)
    axes[1, 0].set_title(f"Hue Mean (Avg: {df['h_mean'].mean():.1f})", fontsize=11, weight="bold")
    axes[1, 0].set_xlabel("Hue Value [0-180]")

    sns.histplot(df["s_mean"], ax=axes[1, 1], color="darkorange", kde=True, bins=30)
    axes[1, 1].set_title(f"Saturation Mean (Avg: {df['s_mean'].mean():.1f})", fontsize=11, weight="bold")
    axes[1, 1].set_xlabel("Saturation Value [0-255]")

    sns.histplot(df["v_mean"], ax=axes[1, 2], color="gray", kde=True, bins=30)
    axes[1, 2].set_title(f"Value / Brightness Mean (Avg: {df['v_mean'].mean():.1f})", fontsize=11, weight="bold")
    axes[1, 2].set_xlabel("Brightness Value [0-255]")

    plt.tight_layout()
    plt.savefig(output_dir / "color_channel_distribution.png", dpi=300)
    plt.close()

    # 6. Sample Images Grid (16 Diverse Classes)
    sample_classes = sorted(df["class"].unique())[:16]
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    axes = axes.flatten()

    for idx, c in enumerate(sample_classes):
        class_df = df[df["class"] == c]
        if len(class_df) > 0:
            sample_row = class_df.iloc[0]
            rgb = read_image_rgb(sample_row["filepath"])
            if rgb is not None:
                axes[idx].imshow(rgb)
                axes[idx].set_title(f"{c.capitalize()}\n({sample_row['width']}x{sample_row['height']})", fontsize=10, weight="bold")
            axes[idx].axis("off")
            
    plt.suptitle("Sample Images from Fruit and Vegetable Classes", fontsize=15, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "sample_classes_grid.png", dpi=300)
    plt.close()

    print("All figures successfully created.")

def run_eda():
    """
    Main function to execute the EDA pipeline and return metrics dictionary.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = collect_image_metadata(DATA_DIR)
    
    # Save raw metadata CSV for reference
    df.to_csv(REPORT_DIR / "dataset_metadata.csv", index=False)
    
    # Generate Visual Figures
    generate_visualizations(df, FIGURES_DIR)
    
    # Print Statistical Summary
    total_imgs = len(df)
    corrupted_count = df["is_corrupt"].sum()
    class_count = df["class"].nunique()
    
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA) COMPLETE")
    print("=" * 60)
    print(f"Total Images Analyzed: {total_imgs}")
    print(f"Corrupted Images:     {corrupted_count}")
    print(f"Total Unique Classes: {class_count}")
    print(f"Splits:               {df['split'].value_counts().to_dict()}")
    print(f"Image Formats:        {df['format'].value_counts().to_dict()}")
    print("-" * 60)
    print("Dimensional Statistics:")
    print(f"  Width:  min={df['width'].min()}, max={df['width'].max()}, mean={df['width'].mean():.1f}, median={df['width'].median():.1f}")
    print(f"  Height: min={df['height'].min()}, max={df['height'].max()}, mean={df['height'].mean():.1f}, median={df['height'].median():.1f}")
    print(f"  Aspect Ratio (W/H): min={df['aspect_ratio'].min():.2f}, max={df['aspect_ratio'].max():.2f}, mean={df['aspect_ratio'].mean():.2f}, median={df['aspect_ratio'].median():.2f}")
    print(f"  File Size (KB):     min={df['file_size_kb'].min():.1f}KB, max={df['file_size_kb'].max():.1f}KB, mean={df['file_size_kb'].mean():.1f}KB")
    print("-" * 60)
    print(f"Detailed metadata saved to: {REPORT_DIR / 'dataset_metadata.csv'}")
    print(f"Analytical figures saved to: {FIGURES_DIR}")
    print("=" * 60 + "\n")
    
    return df

if __name__ == "__main__":
    run_eda()
