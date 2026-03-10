"""
Tests for the Deep Learning Computer Vision framework.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cnn_layers import Conv2D, MaxPool2D, Flatten, Dense, ReLU
from augmentation import ImageAugmentor
from evaluation import ModelEvaluator


class TestConv2D:
    def test_forward_shape(self):
        conv = Conv2D(1, 4, kernel_size=3, padding=1, seed=42)
        x = np.random.randn(2, 1, 8, 8)
        out = conv.forward(x)
        assert out.shape == (2, 4, 8, 8)

    def test_forward_no_padding(self):
        conv = Conv2D(1, 2, kernel_size=3, seed=42)
        x = np.random.randn(1, 1, 6, 6)
        out = conv.forward(x)
        assert out.shape == (1, 2, 4, 4)

    def test_backward_shape(self):
        conv = Conv2D(1, 2, kernel_size=3, padding=1, seed=42)
        x = np.random.randn(1, 1, 6, 6)
        out = conv.forward(x)
        d_out = np.ones_like(out)
        d_input = conv.backward(d_out)
        assert d_input.shape == x.shape


class TestMaxPool2D:
    def test_forward_shape(self):
        pool = MaxPool2D(pool_size=2)
        x = np.random.randn(1, 1, 8, 8)
        out = pool.forward(x)
        assert out.shape == (1, 1, 4, 4)

    def test_max_value(self):
        pool = MaxPool2D(pool_size=2)
        x = np.array([[[[1, 2], [3, 4]]]]).astype(float)
        out = pool.forward(x)
        assert out[0, 0, 0, 0] == 4.0

    def test_backward_shape(self):
        pool = MaxPool2D(pool_size=2)
        x = np.random.randn(1, 1, 4, 4)
        out = pool.forward(x)
        d_out = np.ones_like(out)
        d_input = pool.backward(d_out)
        assert d_input.shape == x.shape


class TestFlatten:
    def test_flatten(self):
        flatten = Flatten()
        x = np.random.randn(2, 3, 4, 4)
        out = flatten.forward(x)
        assert out.shape == (2, 48)

    def test_backward(self):
        flatten = Flatten()
        x = np.random.randn(2, 3, 4, 4)
        flatten.forward(x)
        d_out = np.random.randn(2, 48)
        d_input = flatten.backward(d_out)
        assert d_input.shape == (2, 3, 4, 4)


class TestAugmentation:
    def setup_method(self):
        self.augmentor = ImageAugmentor(seed=42)
        self.image = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)

    def test_horizontal_flip(self):
        flipped = self.augmentor.horizontal_flip(self.image, p=1.0)
        np.testing.assert_array_equal(flipped[:, 0], self.image[:, -1])

    def test_brightness(self):
        bright = self.augmentor.random_brightness(self.image, (1.5, 1.5))
        assert bright.dtype == np.uint8

    def test_random_noise(self):
        noisy = self.augmentor.random_noise(self.image, std=10)
        assert noisy.shape == self.image.shape

    def test_random_crop(self):
        cropped = self.augmentor.random_crop(self.image, 8, 8)
        assert cropped.shape == (8, 8, 3)

    def test_center_crop(self):
        cropped = self.augmentor.center_crop(self.image, 10, 10)
        assert cropped.shape == (10, 10, 3)

    def test_normalize(self):
        norm = self.augmentor.normalize(self.image)
        assert norm.max() <= 1.0

    def test_augment_chain(self):
        result = self.augmentor.augment(self.image)
        assert result.shape == self.image.shape

    def test_generate_batch(self):
        images = [self.image, self.image]
        batch = self.augmentor.generate_batch(images, augmentations_per_image=2)
        assert len(batch) == 6  # 2 originals + 2*2 augmented


class TestEvaluation:
    def setup_method(self):
        self.evaluator = ModelEvaluator()

    def test_confusion_matrix(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1])
        cm = self.evaluator.confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)
        assert cm[0, 0] == 2
        assert cm[1, 2] == 1

    def test_accuracy(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        assert self.evaluator.accuracy(y_true, y_pred) == 1.0

    def test_precision_recall_f1(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        metrics = self.evaluator.precision_recall_f1(y_true, y_pred)
        assert 0 in metrics
        assert 1 in metrics
        assert metrics[1]["recall"] == 1.0

    def test_classification_report(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        report = self.evaluator.classification_report(y_true, y_pred, ["cat", "dog"])
        assert "cat" in report
        assert "dog" in report
        assert "Accuracy" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
