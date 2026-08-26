"""
Demonstration and Evaluation Script for Image Preprocessing Pipeline.
Processes sample fruit and vegetable images, generates comparison figures,
and saves visual reports.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Ensure UTF-8 output for Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from preprocessing import (
    load_image_rgb, letterbox_resize, apply_clahe_lab, 
    denoise_image, extract_canny_edges, normalize_tensor, ImagePreprocessor
)
from augmentation import DataAugmenter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports" / "preprocessing"
FIGURES_DIR = REPORT_DIR / "figures"

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Arial"

def find_sample_images(data_dir: Path):
    """Finds representative sample images across different classes."""
    train_dir = data_dir / "train"
    samples = {}
    target_classes = ["apple", "banana", "carrot", "bell pepper", "spinach", "watermelon"]
    
    for tc in target_classes:
        class_folder = train_dir / tc
        if class_folder.exists():
            imgs = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.jpeg")) + list(class_folder.glob("*.png"))
            if imgs:
                samples[tc] = imgs[0]
    return samples

def generate_preprocessing_figures():
    """Generates all comparative and analytical figures for the report."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating preprocessing analytical figures in '{FIGURES_DIR}'...")

    samples = find_sample_images(DATA_DIR)
    if not samples:
        print("Error: No sample images found in data directory.")
        return

    # Sample for detailed demo (e.g. Carrot or Banana with elongated aspect ratio)
    elongated_sample = samples.get("carrot", list(samples.values())[0])
    color_sample = samples.get("bell pepper", list(samples.values())[0])
    texture_sample = samples.get("spinach", list(samples.values())[0])

    elongated_rgb = load_image_rgb(elongated_sample)
    color_rgb = load_image_rgb(color_sample)
    texture_rgb = load_image_rgb(texture_sample)

    # 1. Resizing & Letterboxing Comparison
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    
    # Original
    axes[0].imshow(elongated_rgb)
    axes[0].set_title(f"Original Image\n({elongated_rgb.shape[1]}x{elongated_rgb.shape[0]})", fontsize=11, weight="bold")
    axes[0].axis("off")

    # Direct Stretch Resize (distorts morphology)
    stretched = cv2.resize(elongated_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    axes[1].imshow(stretched)
    axes[1].set_title("Direct Stretch Resize (224x224)\n[Morphological Distortion!]", fontsize=11, weight="bold", color="darkred")
    axes[1].axis("off")

    # Letterbox Constant Padding
    letterboxed_const, _ = letterbox_resize(elongated_rgb, target_size=(224, 224), pad_mode="constant")
    axes[2].imshow(letterboxed_const)
    axes[2].set_title("Letterbox + Zero Padding (224x224)\n[Preserves Aspect Ratio]", fontsize=11, weight="bold", color="darkgreen")
    axes[2].axis("off")

    # Letterbox Reflect Padding
    letterboxed_refl, _ = letterbox_resize(elongated_rgb, target_size=(224, 224), pad_mode="reflect")
    axes[3].imshow(letterboxed_refl)
    axes[3].set_title("Letterbox + Reflect Padding (224x224)\n[Boundary Preservation]", fontsize=11, weight="bold", color="navy")
    axes[3].axis("off")

    plt.suptitle("Comparison: Direct Resizing vs Aspect-Ratio Preserving Letterboxing", fontsize=13, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "letterbox_vs_stretch.png", dpi=300)
    plt.close()

    # 2. CLAHE Color Enhancement in LAB space
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    # Raw
    axes[0].imshow(color_rgb)
    axes[0].set_title(f"Raw Input (RGB)\n({color_sample.parent.name})", fontsize=11, weight="bold")
    axes[0].axis("off")

    # Grayscale
    gray = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2GRAY)
    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Grayscale Conversion\n[Loses Chromatic Signatures]", fontsize=11, weight="bold")
    axes[1].axis("off")

    # Global Histogram Equalization (Distorts Hue)
    yuv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    global_eq = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    axes[2].imshow(global_eq)
    axes[2].set_title("Global Hist Equalization\n[Prone to Over-saturation]", fontsize=11, weight="bold")
    axes[2].axis("off")

    # CLAHE in CIE-LAB space (Recommended)
    clahe_enhanced = apply_clahe_lab(color_rgb, clip_limit=2.0)
    axes[3].imshow(clahe_enhanced)
    axes[3].set_title("CLAHE on L-channel in CIE-LAB\n[Optimal Contrast & True Hues]", fontsize=11, weight="bold", color="darkgreen")
    axes[3].axis("off")

    plt.suptitle("Illumination Equalization & Contrast Enhancement in Color Spaces", fontsize=13, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "clahe_illumination_enhancement.png", dpi=300)
    plt.close()

    # 3. Denoising and Edge Extraction
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    resized_texture, _ = letterbox_resize(texture_rgb, (224, 224))
    
    # Original Resized
    axes[0].imshow(resized_texture)
    axes[0].set_title(f"Input ({texture_sample.parent.name})\n(224x224)", fontsize=11, weight="bold")
    axes[0].axis("off")

    # Gaussian Blur
    gauss = cv2.GaussianBlur(resized_texture, (5, 5), 0)
    axes[1].imshow(gauss)
    axes[1].set_title("Gaussian Blur\n[Blurs Edges]", fontsize=11, weight="bold")
    axes[1].axis("off")

    # Bilateral Filter (Edge-preserving)
    bilateral = cv2.bilateralFilter(resized_texture, d=7, sigmaColor=50, sigmaSpace=50)
    axes[2].imshow(bilateral)
    axes[2].set_title("Bilateral Filter\n[Denoised + Sharp Edges]", fontsize=11, weight="bold", color="darkgreen")
    axes[2].axis("off")

    # Canny Edge Extraction
    edges = extract_canny_edges(bilateral)
    axes[3].imshow(edges, cmap="magma")
    axes[3].set_title("Canny Edge Map\n[Structural Boundary Features]", fontsize=11, weight="bold", color="navy")
    axes[3].axis("off")

    plt.suptitle("Denoising Filters & Morphological Edge Extraction", fontsize=13, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "denoise_and_edge_detection.png", dpi=300)
    plt.close()

    # 4. Data Augmentation Showcase
    augmenter = DataAugmenter()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    sample_img, _ = letterbox_resize(load_image_rgb(samples.get("apple", list(samples.values())[0])), (224, 224))
    axes[0].imshow(sample_img)
    axes[0].set_title("Base Input (Apple)", fontsize=11, weight="bold", color="darkgreen")
    axes[0].axis("off")

    for i in range(1, 8):
        aug_img = augmenter.augment(sample_img)
        axes[i].imshow(aug_img)
        axes[i].set_title(f"Augmented Variation #{i}", fontsize=10, weight="bold")
        axes[i].axis("off")

    plt.suptitle("Stochastic Data Augmentation Pipeline (Geometric + Photometric)", fontsize=13, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "augmentation_showcase.png", dpi=300)
    plt.close()

    # 5. End-to-End Pipeline Visualization
    pipeline = ImagePreprocessor(target_size=(224, 224), enable_clahe=True, enable_denoising=True)
    res = pipeline.process(elongated_sample)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(res["raw_rgb"])
    axes[0].set_title("Step 1: Raw Image", fontsize=11, weight="bold")
    axes[0].axis("off")

    axes[1].imshow(res["resized_rgb"])
    axes[1].set_title("Step 2: Letterbox Resize (224x224)", fontsize=11, weight="bold")
    axes[1].axis("off")

    axes[2].imshow(res["denoised_rgb"])
    axes[2].set_title("Step 3: Bilateral Denoising", fontsize=11, weight="bold")
    axes[2].axis("off")

    axes[3].imshow(res["enhanced_rgb"])
    axes[3].set_title("Step 4: LAB CLAHE Contrast", fontsize=11, weight="bold")
    axes[3].axis("off")

    # Heatmap of normalized tensor
    norm_vis = np.clip(res["normalized_tensor"] * 0.229 + 0.485, 0.0, 1.0)
    axes[4].imshow(norm_vis)
    axes[4].set_title("Step 5: Z-Score Normalized Tensor", fontsize=11, weight="bold", color="darkgreen")
    axes[4].axis("off")

    plt.suptitle("End-to-End Preprocessing Transformation Workflow", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "end_to_end_preprocessed_pipeline.png", dpi=300)
    plt.close()

    print("All preprocessing demonstration figures generated successfully.")

if __name__ == "__main__":
    generate_preprocessing_figures()
