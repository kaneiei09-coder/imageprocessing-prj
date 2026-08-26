"""
Data Augmentation Module for Fruit and Vegetable Image Recognition.
Provides stochastic geometric and photometric transformations to enhance dataset diversity
and prevent overfitting on minority classes.
"""

import random
from typing import Tuple, Optional
import cv2
import numpy as np

class DataAugmenter:
    """
    Applies configurable randomized data augmentation to RGB images.
    """
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-20.0, 20.0),
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.2,
        zoom_range: Tuple[float, float] = (0.9, 1.1),
        brightness_range: Tuple[float, float] = (-25.0, 25.0),
        contrast_range: Tuple[float, float] = (0.85, 1.15),
        saturation_range: Tuple[float, float] = (0.85, 1.15)
    ):
        self.rotation_range = rotation_range
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.zoom_range = zoom_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range

    def random_rotate(self, image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        """Rotates image around center."""
        if angle is None:
            angle = random.uniform(*self.rotation_range)
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    def random_flip(self, image: np.ndarray) -> np.ndarray:
        """Randomly flips image horizontally or vertically."""
        img = image.copy()
        if random.random() < self.hflip_prob:
            img = cv2.flip(img, 1)  # Horizontal
        if random.random() < self.vflip_prob:
            img = cv2.flip(img, 0)  # Vertical
        return img

    def random_zoom(self, image: np.ndarray, zoom_factor: Optional[float] = None) -> np.ndarray:
        """Randomly zooms into or out of image."""
        if zoom_factor is None:
            zoom_factor = random.uniform(*self.zoom_range)
        h, w = image.shape[:2]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if zoom_factor >= 1.0:
            # Crop center
            top = (new_h - h) // 2
            left = (new_w - w) // 2
            return resized[top:top + h, left:left + w]
        else:
            # Pad center
            pad_top = (h - new_h) // 2
            pad_bottom = h - new_h - pad_top
            pad_left = (w - new_w) // 2
            pad_right = w - new_w - pad_left
            return cv2.copyMakeBorder(
                resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
            )

    def random_photometric(self, image: np.ndarray) -> np.ndarray:
        """Applies random brightness, contrast, and saturation jitter."""
        img_float = image.astype(np.float32)

        # Contrast and Brightness
        alpha = random.uniform(*self.contrast_range)
        beta = random.uniform(*self.brightness_range)
        img_float = np.clip(alpha * img_float + beta, 0, 255).astype(np.uint8)

        # Saturation in HSV
        sat_factor = random.uniform(*self.saturation_range)
        hsv = cv2.cvtColor(img_float, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def augment(self, image: np.ndarray) -> np.ndarray:
        """Applies full augmentation pipeline."""
        img = self.random_rotate(image)
        img = self.random_zoom(img)
        img = self.random_flip(img)
        img = self.random_photometric(img)
        return img

if __name__ == "__main__":
    print("Data Augmenter Module initialized.")
