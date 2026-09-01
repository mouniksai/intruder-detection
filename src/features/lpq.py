"""
lpq.py
======
Pipeline 2: LPQ (Local Phase Quantization) Extractor

Theoretical Formulation:
-------------------------
Local Phase Quantization (LPQ) was introduced by Ojansivu and Heikkilä (2008) for texture
classification under centrally symmetric blur (motion, out-of-focus, atmospheric turbulence).

Mathematical Process:
1. 2D Short-Time Fourier Transform (STFT):
   For each pixel position x = (r, c), a rectangular neighborhood N_x of size win_size x win_size
   is analyzed in the frequency domain at 4 low frequencies:
       w_1 = [a, 0]^T,  w_2 = [0, a]^T,  w_3 = [a, a]^T,  w_4 = [a, -a]^T
   where a = 1 / win_size.

   The STFT coefficients are computed via 1D separable convolutions with basis functions:
       w_x(u) = exp(-j * 2 * pi * a * u)

2. Complex Phase Representation:
   For the 4 frequencies, the real and imaginary parts are separated:
       F(x) = [Re(F_1), Im(F_1), Re(F_2), Im(F_2), Re(F_3), Im(F_3), Re(F_4), Im(F_4)]^T in R^8

3. Decorrelation & Whitening:
   A linear transformation V (derived from SVD/PCA over the covariance of F) decorrelates the 8 components:
       G(x) = V^T * F(x)

4. Binary Quantization:
   Each component of G(x) is quantized using a scalar threshold at 0:
       q_j(x) = 1 if G_j(x) >= 0 else 0

5. 8-Bit Integer Encoding:
   The 8 binary outcomes form an 8-bit integer value in [0, 255]:
       LPQ_code(x) = sum_{j=0}^{7} q_j(x) * 2^j

6. Spatial Multi-Block Histogram:
   The face image is split into B_y x B_x spatial blocks (e.g. 4x4).
   An 8-bit histogram (256 bins) is computed per block, L1-normalized, concatenated,
   and finally L2-normalized into the global feature vector.

Output Vector Length:
--------------------
For 4x4 spatial blocks: 16 blocks * 256 bins = 4096 dimensions.
"""

from typing import Tuple, Optional
import numpy as np
import scipy.signal


class LPQExtractor:
    """
    Local Phase Quantization (LPQ) descriptor for Pipeline 2.
    """

    def __init__(
        self,
        win_size: int = 7,
        freq_estimation_param: float = 1.0,
        grid_size: Tuple[int, int] = (4, 4),
        decorrelate: bool = True
    ) -> None:
        """
        Initialize the LPQ Extractor.

        Args:
            win_size: Local window filter size (must be odd, e.g. 7).
            freq_estimation_param: Frequency parameter scalar (default: 1.0 -> a = 1/win_size).
            grid_size: Spatial grid division (rows, cols) for localized histograms.
            decorrelate: Whether to apply whitening/decorrelation to frequency coefficients.
        """
        if win_size % 2 == 0:
            raise ValueError("win_size must be an odd integer.")

        self.win_size = win_size
        self.freq_param = freq_estimation_param
        self.grid_size = grid_size
        self.decorrelate = decorrelate

        # Precompute 1D separable STFT basis filters
        self._precompute_filters()

    def _precompute_filters(self) -> None:
        """
        Precompute separable 1D convolution kernels for 2D STFT evaluation at 4 frequencies.
        """
        r = (self.win_size - 1) // 2
        u = np.arange(-r, r + 1)
        a = self.freq_param / self.win_size

        # 1D basis vectors
        # w0: constant (dc)
        w0 = np.ones_like(u, dtype=np.float64)
        # w1: complex sinusoid exp(-j*2*pi*a*u)
        w1_re = np.cos(2 * np.pi * a * u)
        w1_im = -np.sin(2 * np.pi * a * u)

        # Build 8 2D spatial filter kernels corresponding to [Re(F1), Im(F1), Re(F2), Im(F2), ...]
        # F1 = w_1 = [a, 0]^T -> w1(x) * w0(y)
        # F2 = w_2 = [0, a]^T -> w0(x) * w1(y)
        # F3 = w_3 = [a, a]^T -> w1(x) * w1(y)
        # F4 = w_4 = [a, -a]^T -> w1(x) * conj(w1(y))

        # We construct real and imaginary filters:
        f_list = []
        # F1 = [a, 0]^T
        f_list.append(np.outer(w0, w1_re))      # Re(F1)
        f_list.append(np.outer(w0, w1_im))      # Im(F1)

        # F2 = [0, a]^T
        f_list.append(np.outer(w1_re, w0))      # Re(F2)
        f_list.append(np.outer(w1_im, w0))      # Im(F2)

        # F3 = [a, a]^T -> (w1_re + j*w1_im)_x * (w1_re + j*w1_im)_y
        # Re(F3) = w1_re_x * w1_re_y - w1_im_x * w1_im_y
        # Im(F3) = w1_re_x * w1_im_y + w1_im_x * w1_re_y
        f_list.append(np.outer(w1_re, w1_re) - np.outer(w1_im, w1_im))  # Re(F3)
        f_list.append(np.outer(w1_im, w1_re) + np.outer(w1_re, w1_im))  # Im(F3)

        # F4 = [a, -a]^T -> (w1_re + j*w1_im)_x * (w1_re - j*w1_im)_y
        # Re(F4) = w1_re_x * w1_re_y + w1_im_x * w1_im_y
        # Im(F4) = -w1_re_x * w1_im_y + w1_im_x * w1_re_y
        f_list.append(np.outer(w1_re, w1_re) + np.outer(w1_im, w1_im))  # Re(F4)
        f_list.append(np.outer(w1_im, w1_re) - np.outer(w1_re, w1_im))  # Im(F4)

        self.filter_bank = np.array(f_list, dtype=np.float32)

    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract spatial multi-block LPQ feature vector from a normalized face image.

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
        num_filters = 8
        responses = np.zeros((num_filters, H, W), dtype=np.float32)

        # Compute filter responses for each of the 8 basis functions
        for i in range(num_filters):
            responses[i] = scipy.signal.convolve2d(img, self.filter_bank[i], mode='same', boundary='symm')

        # Reshape to (8, H*W)
        F = responses.reshape((num_filters, -1))

        if self.decorrelate:
            # SVD whitening of the 8-component frequency vector
            cov = np.cov(F)
            # Add regularization for numerical stability
            cov += np.eye(8, dtype=np.float32) * 1e-6
            u, s, vt = np.linalg.svd(cov)
            # Whitening transform V = U
            G = np.dot(u.T, F)
        else:
            G = F

        # Quantize G: positive -> 1, negative -> 0
        B = (G >= 0).astype(np.int32)  # shape (8, H*W)

        # Convert 8 binary bits to 8-bit integer [0, 255]
        powers_of_two = (2 ** np.arange(8, dtype=np.int32))[:, np.newaxis]
        lpq_codes_flat = np.sum(B * powers_of_two, axis=0)
        lpq_code_image = lpq_codes_flat.reshape((H, W))

        # Multi-block spatial histogram computation
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

                block_codes = lpq_code_image[r0:r1, c0:c1]
                hist, _ = np.histogram(block_codes, bins=256, range=(0, 256))
                hist = hist.astype(np.float32)
                sum_h = np.sum(hist)
                if sum_h > 0:
                    hist /= sum_h
                histograms.append(hist)

        feature_vector = np.concatenate(histograms)
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector /= norm

        return feature_vector.astype(np.float32)


def extract_features_lpq(
    face_image: np.ndarray,
    win_size: int = 7,
    grid_size: Tuple[int, int] = (4, 4)
) -> np.ndarray:
    """
    Standard interface function for Pipeline 2 LPQ extractor.

    Args:
        face_image: 2D uint8/float32 face array (128x128).
        win_size: STFT window size (default: 7).
        grid_size: Spatial histogram grid (default: (4, 4)).

    Returns:
        1D feature vector.
    """
    extractor = LPQExtractor(win_size=win_size, grid_size=grid_size)
    return extractor.extract(face_image)
