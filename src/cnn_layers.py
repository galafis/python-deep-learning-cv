"""
CNN Layers Module
Conv2D, MaxPool2D, Flatten, Dense layers for CNNs built from scratch with NumPy.
"""

import numpy as np
from typing import Optional, Tuple


class Conv2D:
    """2D Convolution layer."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 0, seed: Optional[int] = None):
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = rng.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.biases = np.zeros(out_channels)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.input = None
        self.dweights = None
        self.dbiases = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        Args:
            x: Input of shape (batch, channels, height, width).
        Returns:
            Output of shape (batch, out_channels, out_h, out_w).
        """
        self.input = x
        batch, c_in, h, w = x.shape
        out_c = self.weights.shape[0]

        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0),
                           (self.padding, self.padding),
                           (self.padding, self.padding)))

        out_h = (x.shape[2] - self.kernel_size) // self.stride + 1
        out_w = (x.shape[3] - self.kernel_size) // self.stride + 1
        output = np.zeros((batch, out_c, out_h, out_w))

        for b in range(batch):
            for oc in range(out_c):
                for i in range(out_h):
                    for j in range(out_w):
                        si = i * self.stride
                        sj = j * self.stride
                        region = x[b, :, si:si+self.kernel_size, sj:sj+self.kernel_size]
                        output[b, oc, i, j] = np.sum(region * self.weights[oc]) + self.biases[oc]

        return output

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        """Backward pass computing gradients."""
        batch = d_output.shape[0]
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.sum(d_output, axis=(0, 2, 3))

        x = self.input
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0),
                           (self.padding, self.padding),
                           (self.padding, self.padding)))

        d_input = np.zeros_like(x)
        out_h, out_w = d_output.shape[2], d_output.shape[3]

        for b in range(batch):
            for oc in range(self.weights.shape[0]):
                for i in range(out_h):
                    for j in range(out_w):
                        si = i * self.stride
                        sj = j * self.stride
                        self.dweights[oc] += (
                            x[b, :, si:si+self.kernel_size, sj:sj+self.kernel_size]
                            * d_output[b, oc, i, j]
                        )
                        d_input[b, :, si:si+self.kernel_size, sj:sj+self.kernel_size] += (
                            self.weights[oc] * d_output[b, oc, i, j]
                        )

        if self.padding > 0:
            d_input = d_input[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return d_input

    @property
    def parameters(self):
        return [self.weights, self.biases]

    @property
    def gradients(self):
        return [self.dweights, self.dbiases]


class MaxPool2D:
    """2D Max Pooling layer."""

    def __init__(self, pool_size: int = 2, stride: Optional[int] = None):
        self.pool_size = pool_size
        self.stride = stride or pool_size
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch, channels, h, w = x.shape
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1

        output = np.zeros((batch, channels, out_h, out_w))
        self.mask = np.zeros_like(x)

        for b in range(batch):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        si = i * self.stride
                        sj = j * self.stride
                        region = x[b, c, si:si+self.pool_size, sj:sj+self.pool_size]
                        max_val = np.max(region)
                        output[b, c, i, j] = max_val
                        mask_region = (region == max_val)
                        self.mask[b, c, si:si+self.pool_size, sj:sj+self.pool_size] = mask_region

        return output

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        batch, channels, out_h, out_w = d_output.shape
        d_input = np.zeros_like(self.mask)

        for b in range(batch):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        si = i * self.stride
                        sj = j * self.stride
                        d_input[b, c, si:si+self.pool_size, sj:sj+self.pool_size] += (
                            self.mask[b, c, si:si+self.pool_size, sj:sj+self.pool_size]
                            * d_output[b, c, i, j]
                        )
        return d_input


class Flatten:
    """Flatten layer for transitioning from conv to dense."""

    def __init__(self):
        self.input_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        return d_output.reshape(self.input_shape)


class Dense:
    """Fully connected layer."""

    def __init__(self, input_size: int, output_size: int, seed: Optional[int] = None):
        rng = np.random.RandomState(seed)
        self.weights = rng.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        self.input = None
        self.dweights = None
        self.dbiases = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return x @ self.weights + self.biases

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        self.dweights = self.input.T @ d_output
        self.dbiases = np.sum(d_output, axis=0, keepdims=True)
        return d_output @ self.weights.T

    @property
    def parameters(self):
        return [self.weights, self.biases]

    @property
    def gradients(self):
        return [self.dweights, self.dbiases]


class ReLU:
    """ReLU activation for CNN."""

    def __init__(self):
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0, x)

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        return d_output * (self.input > 0).astype(float)
