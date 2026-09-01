"""
gabor.py
========
Pipeline 4: Gabor Wavelet Filter Bank Extractor

Theoretical Formulation:
-------------------------
Gabor wavelets model the spatial frequency and orientation selectivity of simple cells
in the mammalian primary visual cortex (V1) (Daugman, 1985; Liu & Wechsler, IEEE TPAMI 2002).

Mathematical Process:
1. 2D Gabor Filter Definition:
   A Gabor filter psi_{mu, nu}(x, y) is parameterized by orientation mu and scale nu:
       psi_{mu, nu}(x, y) = (||k_{mu, nu}||^2 / sigma^2) * exp(- ||k_{mu, nu}||^2 * (x^2 + y^2) / (2 * sigma^2))
                            * [ exp(j * k_{mu, nu}^T * [x, y]^T) - exp(- sigma^2 / 2) ]
   where:
       - Scales (5 frequencies): nu in {0, 1, 2, 3, 4} -> wavelength lambda_nu = 4 * (sqrt(2))^nu
       - Orientations (8 angles): mu in {0, 1, ..., 7} -> theta_mu = mu * pi / 8
       - sigma = 2 * pi (Gaussian envelope standard deviation)
       Total filter bank size = 5 scales * 8 orientations = 40 Gabor filters.

2. Magnitude Response Maps:
   The face image I(x, y) is convolved with all 40 Gabor kernels:
       O_{mu, nu}(x, y) = I(x, y) * psi_{mu, nu}(x, y)
   The magnitude response represents localized energy at that scale and orientation:
       M_{mu, nu}(x, y) = sqrt( Re(O_{mu, nu})^2 + Im(O_{mu, nu})^2 )

3. Spatial Grid Statistical Pooling:
   The face is partitioned into a B_y x B_x grid (e.g. 4x4 blocks = 16 sub-regions).
   Within each sub-region, 3 statistical descriptors are computed for every filter response:
       - Mean magnitude (average local edge/texture strength): mu_{m,n}
       - Standard deviation (dispersion of local contrast): sigma_{m,n}
       - Energy / L2 norm (total localized frequency power): E_{m,n} = sum(M^2)

Output Vector Length:
--------------------
For 40 filters * (4x4 blocks = 16 regions) * 3 statistics:
Vector dimension = 40 * 16 * 3 = 1920 dimensions.
"""

from typing import Tuple, List, Optional
import numpy as np
import cv2


class GaborExtractor:
    """
    Multiscale Multi-Orientation Gabor Wavelet Bank extractor for Pipeline 4.
    """

    def __init__(
        self,
        num_scales: int = 5,
        num_orientations: int = 8,
        kernel_size: int = 21,
        grid_size: Tuple[int, int] = (4, 4),
        sigma: float = 4.0,
        gamma: float = 0.5
    ) -> None:
        """
        Initialize the 40-filter Gabor Wavelet Bank.

        Args:
            num_scales: Number of spatial frequencies / scales (default: 5).
            num_orientations: Number of orientation directions (default: 8).
            kernel_size: Size of square Gabor kernel (default: 21x21).
            grid_size: Spatial grid division (rows, cols) for statistical pooling.
            sigma: Gaussian envelope bandwidth parameter.
            gamma: Spatial aspect ratio (ellipticity) of Gabor kernel.
        """
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        self.kernel_size = kernel_size
        self.grid_size = grid_size
        self.sigma = sigma
        self.gamma = gamma

        self.kernels: List[Tuple[np.ndarray, np.ndarray]] = []
        self._build_gabor_bank()

    def _build_gabor_bank(self) -> None:
        """
        Construct 40 pairs of Real and Imaginary Gabor filters.
        """
        self.kernels.clear()
        ksize = (self.kernel_size, self.kernel_size)

        # Scale wavelengths from fine to coarse
        lambdas = [3.0 * (1.414 ** s) for s in range(self.num_scales)]

        for s_idx, lmbda in enumerate(lambdas):
            for o_idx in range(self.num_orientations):
                theta = o_idx * np.pi / self.num_orientations

                # Real part (cosine carrier, phase = 0)
                k_re = cv2.getGaborKernel(
                    ksize,
                    sigma=self.sigma * (1.2 ** s_idx),
                    theta=theta,
                    lambd=lmbda,
                    gamma=self.gamma,
                    psi=0,
                    ktype=cv2.CV_32F
                )

                # Imaginary part (sine carrier, phase = pi/2)
                k_im = cv2.getGaborKernel(
                    ksize,
                    sigma=self.sigma * (1.2 ** s_idx),
                    theta=theta,
                    lambd=lmbda,
                    gamma=self.gamma,
                    psi=np.pi / 2.0,
                    ktype=cv2.CV_32F
                )

                # Zero-mean DC correction
                k_re -= np.mean(k_re)
                k_im -= np.mean(k_im)

                self.kernels.append((k_re, k_im))

    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract multi-scale multi-orientation spatial Gabor statistical feature vector.

        Args:
            face_image: 2D Grayscale face image (e.g. 128x128).

        Returns:
            1D float32 normalized feature vector of length (40 * grid_cells * 3).
        """
        if len(face_image.shape) == 3:
            img = face_image[:, :, 0].astype(np.float32)
        else:
            img = face_image.astype(np.float32)

        H, W = img.shape
        grid_rows, grid_cols = self.grid_size
        block_h = H // grid_rows
        block_w = W // grid_cols

        feature_components: List[float] = []

        # Iterate over all 40 Gabor filters
        for k_re, k_im in self.kernels:
            # Convolve image with real and imaginary kernels
            resp_re = cv2.filter2D(img, cv2.CV_32F, k_re, borderType=cv2.BORDER_REFLECT)
            resp_im = cv2.filter2D(img, cv2.CV_32F, k_im, borderType=cv2.BORDER_REFLECT)

            # Compute magnitude response map: sqrt(Re^2 + Im^2)
            magnitude = np.sqrt(resp_re**2 + resp_im**2)

            # Spatial grid pooling over magnitude map
            for r in range(grid_rows):
                for c in range(grid_cols):
                    r0 = r * block_h
                    r1 = (r + 1) * block_h if r < grid_rows - 1 else H
                    c0 = c * block_w
                    c1 = (c + 1) * block_w if c < grid_cols - 1 else W

                    block = magnitude[r0:r1, c0:c1]

                    # Moment 1: Mean magnitude
                    mean_val = float(np.mean(block))
                    # Moment 2: Standard deviation
                    std_val = float(np.std(block))
                    # Moment 3: Energy (normalized sum of squared responses)
                    energy_val = float(np.mean(block**2))

                    feature_components.extend([mean_val, std_val, energy_val])

        feature_vector = np.array(feature_components, dtype=np.float32)

        # Global L2 normalization
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector /= norm

        return feature_vector


def extract_features_gabor(
    face_image: np.ndarray,
    num_scales: int = 5,
    num_orientations: int = 8,
    grid_size: Tuple[int, int] = (4, 4)
) -> np.ndarray:
    """
    Standard interface function for Pipeline 4 Gabor extractor.

    Args:
        face_image: 2D uint8/float32 face array (128x128).
        num_scales: Number of scales (default: 5).
        num_orientations: Number of orientations (default: 8).
        grid_size: Spatial grid division (default: (4, 4)).

    Returns:
        1D feature vector.
    """
    extractor = GaborExtractor(
        num_scales=num_scales,
        num_orientations=num_orientations,
        grid_size=grid_size
    )
    return extractor.extract(face_image)
