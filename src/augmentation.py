"""
Image Augmentation Pipeline
Data augmentation operations for training deep learning models.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class ImageAugmentor:
    """Image augmentation pipeline for data augmentation."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def horizontal_flip(self, image: np.ndarray, p: float = 0.5) -> np.ndarray:
        """Randomly flip image horizontally."""
        if self.rng.random() < p:
            return image[:, ::-1].copy()
        return image

    def vertical_flip(self, image: np.ndarray, p: float = 0.5) -> np.ndarray:
        """Randomly flip image vertically."""
        if self.rng.random() < p:
            return image[::-1, :].copy()
        return image

    def random_brightness(self, image: np.ndarray, factor_range: Tuple[float, float] = (0.7, 1.3)) -> np.ndarray:
        """Randomly adjust brightness."""
        factor = self.rng.uniform(factor_range[0], factor_range[1])
        result = image.astype(np.float64) * factor
        return np.clip(result, 0, 255).astype(np.uint8)

    def random_noise(self, image: np.ndarray, std: float = 10.0) -> np.ndarray:
        """Add random Gaussian noise."""
        noise = self.rng.randn(*image.shape) * std
        result = image.astype(np.float64) + noise
        return np.clip(result, 0, 255).astype(np.uint8)

    def random_crop(self, image: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
        """Randomly crop a region from the image."""
        h, w = image.shape[:2]
        if crop_h >= h or crop_w >= w:
            return image
        top = self.rng.randint(0, h - crop_h)
        left = self.rng.randint(0, w - crop_w)
        return image[top:top + crop_h, left:left + crop_w].copy()

    def center_crop(self, image: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
        """Crop from center of image."""
        h, w = image.shape[:2]
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        return image[top:top + crop_h, left:left + crop_w].copy()

    def normalize(self, image: np.ndarray, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
        """Normalize image to given mean and std."""
        result = image.astype(np.float64) / 255.0
        return (result - mean) / (std if std > 0 else 1.0)

    def augment(self, image: np.ndarray, config: Optional[Dict] = None) -> np.ndarray:
        """Apply a chain of augmentations."""
        config = config or {
            "horizontal_flip": 0.5,
            "brightness": (0.8, 1.2),
            "noise_std": 5.0,
        }
        result = image.copy()

        if "horizontal_flip" in config:
            result = self.horizontal_flip(result, config["horizontal_flip"])
        if "vertical_flip" in config:
            result = self.vertical_flip(result, config["vertical_flip"])
        if "brightness" in config:
            result = self.random_brightness(result, config["brightness"])
        if "noise_std" in config:
            result = self.random_noise(result, config["noise_std"])

        return result

    def generate_batch(self, images: List[np.ndarray], augmentations_per_image: int = 3) -> List[np.ndarray]:
        """Generate augmented batch from list of images."""
        augmented = []
        for img in images:
            augmented.append(img)
            for _ in range(augmentations_per_image):
                augmented.append(self.augment(img))
        return augmented
