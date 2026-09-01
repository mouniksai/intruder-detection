"""
wld.py
======
Pipeline 3: WLD (Weber Local Descriptor) Extractor

Theoretical Formulation:
-------------------------
Weber Local Descriptor (WLD) was introduced by Chen et al. (IEEE TPAMI 2010), inspired by
Weber's Law of psycho-physics, which states that human sensory perception responds to relative
changes in stimulus rather than absolute physical magnitudes (Delta_I / I = constant).

Mathematical Process:
1. Differential Excitation (xi):
   For each central pixel x_c and its p neighbors {x_0, x_1, ..., x_{p-1}} on a radius R:
       Delta_I = sum_{i=0}^{p-1} (x_i - x_c)
       xi(x_c) = arctan[ Delta_I / (x_c + eps) ]
   where xi(x_c) in [-pi/2, pi/2].
   - xi > 0: Central pixel is darker than surrounding neighborhood (valley / spot).
   - xi < 0: Central pixel is brighter than surrounding neighborhood (peak / highlight).
   - xi ~= 0: Homogeneous region.

2. Gradient Orientation (theta):
   Computed via orthogonal directional filters over the 8-neighborhood:
       v_00 = x_7 - x_3  (vertical gradient)
       v_01 = x_5 - x_1  (horizontal gradient)
       theta(x_c) = arctan2(v_00, v_01) in [-pi, pi]
   The continuous angle is quantized into T dominant directional bins:
       Phi_t = (2 * pi * t) / T, for t = 0, ..., T-1

3. 2D Joint Histogram Encoding:
   The feature space combines differential excitation (contrast intensity) and orientation:
   - Orientation space is divided into T sub-regions.
   - For each orientation bin t, differential excitation xi is binned into M intervals,
     each possessing S fine-grained sub-bins.
   - Spatial multi-block partitioning (e.g. 4x4 blocks) preserves topographic face structure.

Output Vector Length:
--------------------
For 4x4 spatial blocks, T=8 orientations, M=6 intervals:
Vector dimension = 4x4 * (8 * 6 * 4) = 3072 dimensions (or calibrated joint histogram).
"""

from typing import Tuple, Optional
import numpy as np
import scipy.ndimage


class WLDExtractor:
    """
    Weber Local Descriptor (WLD) extractor for Pipeline 3.
    """

    def __init__(
        self,
        num_orientations: int = 8,
        num_excitation_bins: int = 32,
        grid_size: Tuple[int, int] = (4, 4),
        epsilon: float = 1e-3
    ) -> None:
        """
        Initialize WLD extractor.

        Args:
            num_orientations: Number of gradient orientation quantization bins T (default: 8).
            num_excitation_bins: Number of differential excitation histogram bins (default: 32).
            grid_size: Spatial grid division (rows, cols) for localized histograms.
            epsilon: Small constant to avoid division by zero.
        """
        self.num_orientations = num_orientations
        self.num_excitation_bins = num_excitation_bins
        self.grid_size = grid_size
        self.epsilon = epsilon

    def _compute_differential_excitation(self, img: np.ndarray) -> np.ndarray:
        """
        Compute differential excitation xi(x_c) for every pixel in the image.
        Uses an 8-neighborhood kernel:
            [-1, -1, -1]
            [-1,  8, -1]  -> sum(x_i - x_c) = sum(x_i) - 8*x_c
        """
        # Kernel computing sum_{i=0}^7 (x_i - x_c) = sum(x_i) - 8*x_c
        kernel = np.array([
            [1.0, 1.0, 1.0],
            [1.0, -8.0, 1.0],
            [1.0, 1.0, 1.0]
        ], dtype=np.float32)

        # Convolve to get Delta_I
        delta_i = scipy.ndimage.convolve(img, kernel, mode='nearest')

        # Compute xi = arctan(delta_i / (img + eps))
        xi = np.arctan(delta_i / (img + self.epsilon))  # Range: [-pi/2, pi/2]
        return xi.astype(np.float32)

    def _compute_gradient_orientation(self, img: np.ndarray) -> np.ndarray:
        """
        Compute gradient orientation theta(x_c) quantized into T orientation bins.
        """
        # Vertical gradient kernel (x7 - x3)
        kernel_v = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0]
        ], dtype=np.float32)

        # Horizontal gradient kernel (x5 - x1)
        kernel_h = np.array([
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0]
        ], dtype=np.float32)

        grad_v = scipy.ndimage.convolve(img, kernel_v, mode='nearest')
        grad_h = scipy.ndimage.convolve(img, kernel_h, mode='nearest')

        # Angle theta in [-pi, pi]
        theta = np.arctan2(grad_v, grad_h)

        # Map [-pi, pi] to [0, 2*pi)
        theta_pos = np.mod(theta + 2 * np.pi, 2 * np.pi)

        # Quantize to [0, num_orientations - 1]
        bin_width = (2 * np.pi) / self.num_orientations
        quantized_theta = np.floor(theta_pos / bin_width).astype(np.int32)
        quantized_theta = np.clip(quantized_theta, 0, self.num_orientations - 1)
        return quantized_theta

    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract spatial multi-block joint WLD feature vector from a normalized face image.

        Args:
            face_image: 2D Grayscale face image (e.g. 128x128).

        Returns:
            1D float32 normalized feature vector.
        """
        if len(face_image.shape) == 3:
            img = face_image[:, :, 0].astype(np.float32)
        else:
            img = face_image.astype(np.float32)

        H, W = img.shape

        # 1. Differential Excitation xi: [-pi/2, pi/2]
        xi = self._compute_differential_excitation(img)

        # 2. Gradient Orientation theta: integer in [0, T-1]
        theta = self._compute_gradient_orientation(img)

        # Multi-block spatial partitioning
        grid_rows, grid_cols = self.grid_size
        block_h = H // grid_rows
        block_w = W // grid_cols
        histograms = []

        xi_min, xi_max = -np.pi / 2.0, np.pi / 2.0

        for r in range(grid_rows):
            for c in range(grid_cols):
                r0 = r * block_h
                r1 = (r + 1) * block_h if r < grid_rows - 1 else H
                c0 = c * block_w
                c1 = (c + 1) * block_w if c < grid_cols - 1 else W

                block_xi = xi[r0:r1, c0:c1]
                block_theta = theta[r0:r1, c0:c1]

                block_features = []
                # Compute excitation histogram for each discrete orientation
                for t in range(self.num_orientations):
                    mask = (block_theta == t)
                    xi_in_bin = block_xi[mask]

                    if len(xi_in_bin) > 0:
                        h, _ = np.histogram(
                            xi_in_bin,
                            bins=self.num_excitation_bins,
                            range=(xi_min, xi_max)
                        )
                        h = h.astype(np.float32)
                    else:
                        h = np.zeros(self.num_excitation_bins, dtype=np.float32)

                    block_features.append(h)

                block_vec = np.concatenate(block_features)
                # L1 normalize block vector
                sum_b = np.sum(block_vec)
                if sum_b > 0:
                    block_vec /= sum_b
                histograms.append(block_vec)

        feature_vector = np.concatenate(histograms)
        # L2 normalize global feature vector
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector /= norm

        return feature_vector.astype(np.float32)


def extract_features_wld(
    face_image: np.ndarray,
    num_orientations: int = 8,
    num_excitation_bins: int = 16,
    grid_size: Tuple[int, int] = (4, 4)
) -> np.ndarray:
    """
    Standard interface function for Pipeline 3 WLD extractor.

    Args:
        face_image: 2D uint8/float32 face array (128x128).
        num_orientations: Orientation bins (default: 8).
        num_excitation_bins: Differential excitation bins (default: 16).
        grid_size: Spatial histogram grid (default: (4, 4)).

    Returns:
        1D feature vector.
    """
    extractor = WLDExtractor(
        num_orientations=num_orientations,
        num_excitation_bins=num_excitation_bins,
        grid_size=grid_size
    )
    return extractor.extract(face_image)
