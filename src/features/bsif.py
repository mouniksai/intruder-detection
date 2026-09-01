"""
bsif.py
=======
Pipeline 1: BSIF (Binarized Statistical Image Features) Extractor

Theoretical Formulation:
-------------------------
BSIF was introduced by Kannala and Rahtu (2012) as a local descriptor inspired by LBP and LPQ,
but substituting heuristic thresholding rules with statistical filter banks learned from natural
image patches using Independent Component Analysis (ICA).

Mathematical Process:
1. Filter Bank:
   Given an image patch I(x, y) and n linear filters {W_1, W_2, ..., W_n} of size l x l:
   The filter response at spatial location (x, y) is:
       s_i(x, y) = (I * W_i)(x, y) = sum_{u, v} W_i(u, v) * I(x - u, y - v)

2. Binarization:
   Each filter response is binarized around zero:
       b_i(x, y) = 1 if s_i(x, y) > 0 else 0

3. Bit-String Integer Encoding:
   The n bits are combined into an integer code in range [0, 2^n - 1]:
       code(x, y) = sum_{i=0}^{n-1} b_i(x, y) * 2^i

4. Spatial Multi-Block Histogram:
   The face image is subdivided into a grid of B_y x B_x sub-regions (e.g. 4x4 blocks).
   A normalized histogram of 2^n bins is computed for each sub-region to preserve
   coarse spatial topography (forehead, eyes, nose, cheeks, mouth).
   The histograms are concatenated into the final feature vector.

Output Vector Length:
--------------------
For n = 8 bits (256 bins) and 4x4 spatial blocks (16 blocks):
Length = 16 * 256 = 4096 dimensions (or 8 blocks -> 2048 dimensions).
"""

from typing import Tuple, Optional
import numpy as np
import scipy.ndimage
from scipy.signal import convolve2d


class BSIFExtractor:
    """
    Binarized Statistical Image Features (BSIF) descriptor for Pipeline 1.
    """

    def __init__(
        self,
        num_bits: int = 8,
        filter_size: int = 7,
        grid_size: Tuple[int, int] = (4, 4),
        random_seed: int = 42
    ) -> None:
        """
        Initialize BSIF extractor.

        Args:
            num_bits: Number of statistical filters (e.g. 8 bits -> 256 histogram bins).
            filter_size: Spatial dimension l of the square filters (l x l).
            grid_size: Spatial grid division (rows, cols) for localized histograms.
            random_seed: Seed for deterministic statistical basis generation.
        """
        self.num_bits = num_bits
        self.filter_size = filter_size
        self.grid_size = grid_size
        self.filters = self._generate_statistical_filters(num_bits, filter_size, random_seed)

    def _generate_statistical_filters(
        self,
        num_filters: int,
        size: int,
        seed: int
    ) -> np.ndarray:
        """
        Generate an orthonormal set of zero-mean statistical ICA-like filters
        derived via Gram-Schmidt orthogonalization over multiscale Gaussian-frequency bases.

        Args:
            num_filters: Number of filters to generate (n).
            size: Dimension of square filter (l x l).
            seed: Seed for reproducibility.

        Returns:
            np.ndarray of shape (num_filters, size, size).
        """
        rng = np.random.RandomState(seed)
        num_pixels = size * size

        # Generate a set of harmonic and wave-like basis functions
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, y)

        raw_filters = []
        # Construct directional derivative and harmonic basis
        for k in range(num_filters):
            freq_x = (k % 3 + 1) * np.pi
            freq_y = ((k // 3) % 3 + 1) * np.pi
            phase = (k * np.pi) / num_filters
            pattern = np.sin(freq_x * xx + freq_y * yy + phase) * np.exp(-(xx**2 + yy**2) / 0.8)
            # Add stochastic component to simulate natural patch statistical divergence
            noise = rng.normal(0, 0.1, (size, size))
            f = pattern + noise
            f -= np.mean(f)  # Zero mean
            raw_filters.append(f.flatten())

        raw_matrix = np.array(raw_filters)  # (num_filters, num_pixels)

        # Orthonormalize via QR decomposition (Gram-Schmidt)
        q, _ = np.linalg.qr(raw_matrix.T)
        ortho_filters = q[:, :num_filters].T  # (num_filters, num_pixels)

        # Reshape to (num_filters, size, size)
        filters = ortho_filters.reshape((num_filters, size, size))
        return filters.astype(np.float32)

    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract spatial multi-block BSIF feature vector from a normalized face image.

        Args:
            face_image: 2D Grayscale face image (e.g. 128x128).

        Returns:
            1D float32 normalized feature vector.
        """
        # Ensure image is single-channel float32
        if len(face_image.shape) == 3:
            img = face_image[:, :, 0].astype(np.float32)
        else:
            img = face_image.astype(np.float32)

        H, W = img.shape
        num_bins = 2 ** self.num_bits
        code_image = np.zeros((H, W), dtype=np.int32)

        # Convolve image with each statistical filter and accumulate bit powers
        for i in range(self.num_bits):
            # 2D Convolution with replicate boundary handling
            filt = self.filters[i]
            response = scipy.ndimage.convolve(img, filt, mode='nearest')
            # Binary indicator (response > 0)
            binary_bit = (response > 0).astype(np.int32)
            code_image += binary_bit * (2 ** i)

        # Multi-block spatial histogram pooling
        grid_rows, grid_cols = self.grid_size
        block_h = H // grid_rows
        block_w = W // grid_cols
        histograms = []

        for r in range(grid_rows):
            for c in range(grid_cols):
                r0 = r * block_h
                r1 = (r + 1) * block_h if r < grid_rows - 1 else H
                c0 = c * block_w
                c1 = (c + 1) * block_w if c < grid_cols - 1 else W

                block_codes = code_image[r0:r1, c0:c1]
                # Compute histogram of codes in this block
                hist, _ = np.histogram(block_codes, bins=num_bins, range=(0, num_bins))
                # L1 normalization per block (with epsilon to avoid division by zero)
                hist = hist.astype(np.float32)
                sum_h = np.sum(hist)
                if sum_h > 0:
                    hist /= sum_h
                histograms.append(hist)

        # Concatenate all spatial block histograms into a global feature vector
        feature_vector = np.concatenate(histograms)
        # L2 normalize global feature vector
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector /= norm

        return feature_vector.astype(np.float32)


def extract_features_bsif(
    face_image: np.ndarray,
    num_bits: int = 8,
    filter_size: int = 7,
    grid_size: Tuple[int, int] = (4, 4)
) -> np.ndarray:
    """
    Standard interface function for Pipeline 1 BSIF extractor.

    Args:
        face_image: 2D uint8/float32 face array (128x128).
        num_bits: Bit depth (default: 8).
        filter_size: Filter size (default: 7).
        grid_size: Spatial histogram grid (default: (4, 4)).

    Returns:
        1D feature vector.
    """
    extractor = BSIFExtractor(num_bits=num_bits, filter_size=filter_size, grid_size=grid_size)
    return extractor.extract(face_image)
