"""
Fruit and Vegetable Image Recognition Dataset Downloader
Downloads dataset from Kaggle using kagglehub and organizes it into the project data directory.
"""

import sys
import os
import shutil
from pathlib import Path

# Ensure UTF-8 output for Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import kagglehub

DATASET_HANDLE = "kritikseth/fruit-and-vegetable-image-recognition"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET_DIR = PROJECT_ROOT / "data"

def download_and_setup_dataset(target_dir: Path = DEFAULT_TARGET_DIR) -> Path:
    """
    Downloads the dataset from Kaggle and copies it into the project data directory.
    """
    print(f"Downloading dataset '{DATASET_HANDLE}' from Kaggle...")
    download_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    print(f"Dataset downloaded to cache: {download_path}")

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if target already has contents
    subdirs = ["train", "test", "validation"]
    existing = [s for s in subdirs if (target_dir / s).exists()]
    
    if len(existing) == len(subdirs):
        print(f"Dataset folders {subdirs} already present in target directory.")
    else:
        print(f"Copying dataset from cache to target directory...")
        for item in download_path.iterdir():
            dest_item = target_dir / item.name
            if item.is_dir():
                if dest_item.exists():
                    shutil.rmtree(dest_item)
                shutil.copytree(item, dest_item)
            else:
                shutil.copy2(item, dest_item)
        print("Copy completed successfully.")

    # Print summary statistics
    print_dataset_summary(target_dir)
    return target_dir

def print_dataset_summary(target_dir: Path):
    """
    Prints the summary of the dataset (splits, categories, image counts).
    """
    print("\n" + "=" * 50)
    print("Dataset Summary:")
    print("=" * 50)
    
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    total_images = 0
    
    for split in ["train", "validation", "test"]:
        split_path = target_dir / split
        if not split_path.exists():
            continue
        
        categories = [d for d in split_path.iterdir() if d.is_dir()]
        split_image_count = 0
        for cat in categories:
            images = [f for f in cat.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
            split_image_count += len(images)
            
        print(f"  [{split.upper():<10}] - {len(categories)} classes, {split_image_count} images")
        total_images += split_image_count

    print("-" * 50)
    print(f"Total Images: {total_images}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    download_and_setup_dataset()
