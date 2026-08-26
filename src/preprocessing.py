"""
Image Preprocessing Pipeline for Fruit and Vegetable Image Recognition.
Includes aspect-ratio preserving letterboxing, CLAHE color enhancement in LAB space,
denoising, edge extraction, and Z-score / Min-Max normalization.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional, Union
import cv2
import numpy as np
from PIL import Image

# Ensure UTF-8 output for Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def load_image_rgb(image_source: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    Loads an image file into an RGB numpy array uint8, robust against Windows Unicode paths.
    """
    if isinstance(image_source, np.ndarray):
        if image_source.ndim == 2:
            return cv2.cvtColor(image_source, cv2.COLOR_GRAY2RGB)
        return image_source.copy()
        
    with Image.open(image_source) as pil_img:
        return np.array(pil_img.convert("RGB"))

def letterbox_resize(
    image: np.ndarray, 
    target_size: Tuple[int, int] = (224, 224), 
    pad_color: Tuple[int, int, int] = (0, 0, 0),
    pad_mode: str = "constant"
) -> Tuple[np.ndarray, dict]:
    """
    Resizes image maintaining original aspect ratio and pads to target_size (W, H).
    Returns (padded_image, transform_info).
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]

    # Calculate scale factor
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Resize using high quality interpolation
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    # Calculate symmetric padding
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top

    if pad_mode == "reflect":
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_REFLECT_101
        )
    else:
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, value=pad_color
        )

    transform_info = {
        "scale": scale,
        "new_size": (new_w, new_h),
        "padding": (pad_top, pad_bottom, pad_left, pad_right),
        "target_size": target_size
    }
    return padded, transform_info

def apply_clahe_lab(
    image_rgb: np.ndarray, 
    clip_limit: float = 2.0, 
    grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel
    in CIE-LAB color space to enhance local contrast without hue shifts.
    """
    # Convert RGB to CIE-LAB
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Create and apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl = clahe.apply(l_channel)

    # Merge channels and convert back to RGB
    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb

def denoise_image(
    image_rgb: np.ndarray, 
    method: str = "bilateral", 
    d: int = 7, 
    sigma_color: float = 50.0, 
    sigma_space: float = 50.0
) -> np.ndarray:
    """
    Applies edge-preserving smoothing (Bilateral) or Gaussian filtering.
    """
    if method == "bilateral":
        # Bilateral filter operates on BGR/RGB
        return cv2.bilateralFilter(image_rgb, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    elif method == "gaussian":
        return cv2.GaussianBlur(image_rgb, (5, 5), 0)
    return image_rgb

def extract_canny_edges(
    image_rgb: np.ndarray, 
    low_thresh: int = 50, 
    high_thresh: int = 150
) -> np.ndarray:
    """
    Extracts Canny edges from image for morphological shape analysis.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    return edges

def normalize_tensor(
    image_rgb: np.ndarray, 
    method: str = "z_score",
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Normalizes image pixel values into float32 range [0, 1] or standard ImageNet Z-score.
    """
    # Scale to [0, 1]
    norm_img = image_rgb.astype(np.float32) / 255.0

    if method == "minmax":
        return norm_img
    elif method == "z_score":
        mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        return (norm_img - mean_arr) / std_arr
    elif method == "centering":
        return (norm_img - 0.5) * 2.0  # Range [-1, 1]
    return norm_img

class ImagePreprocessor:
    """
    Configurable image preprocessing pipeline for model inference and training.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        enable_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        enable_denoising: bool = True,
        denoise_method: str = "bilateral",
        pad_mode: str = "constant",
        normalization_method: str = "z_score"
    ):
        self.target_size = target_size
        self.enable_clahe = enable_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.enable_denoising = enable_denoising
        self.denoise_method = denoise_method
        self.pad_mode = pad_mode
        self.normalization_method = normalization_method

    def process(self, image_source: Union[str, Path, np.ndarray]) -> dict:
        """
        Executes end-to-end preprocessing pipeline on a single image.
        Returns a dictionary containing intermediate and final processed representations.
        """
        raw_rgb = load_image_rgb(image_source)

        # 1. Letterbox Resize (Aspect-ratio preserving)
        resized_rgb, transform_info = letterbox_resize(
            raw_rgb, target_size=self.target_size, pad_mode=self.pad_mode
        )

        # 2. Denoising
        denoised_rgb = resized_rgb
        if self.enable_denoising:
            denoised_rgb = denoise_image(resized_rgb, method=self.denoise_method)

        # 3. CLAHE Contrast Enhancement
        enhanced_rgb = denoised_rgb
        if self.enable_clahe:
            enhanced_rgb = apply_clahe_lab(denoised_rgb, clip_limit=self.clahe_clip_limit)

        # 4. Edge Extraction
        edges = extract_canny_edges(enhanced_rgb)

        # 5. Normalization
        normalized_tensor = normalize_tensor(
            enhanced_rgb, method=self.normalization_method
        )

        return {
            "raw_rgb": raw_rgb,
            "resized_rgb": resized_rgb,
            "denoised_rgb": denoised_rgb,
            "enhanced_rgb": enhanced_rgb,
            "edges": edges,
            "normalized_tensor": normalized_tensor,
            "transform_info": transform_info
        }

if __name__ == "__main__":
    print("Image Preprocessor Module initialized successfully.")
