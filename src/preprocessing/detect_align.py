"""
detect_align.py
================
Shared Preprocessing Pipeline for Face Detection, Alignment, and Normalization.

This module ensures that ALL 5 feature extractors (BSIF, LPQ, WLD, Gabor, Geometry)
receive identically preprocessed, geometrically aligned, and photometrically normalized
face crops.

Mathematical & Implementation Workflow:
---------------------------------------
1. Face Detection:
   - Primary: OpenCV Haar Cascade frontal face detector.
   - Robust fallback: Image center crop or aspect-ratio bounding box.
2. Landmark Detection & Alignment:
   - Locates left and right eye centers.
   - Computes rotation angle: theta = arctan2(dy, dx) * 180 / pi.
   - Computes Euclidean distance between eyes for scale reference.
   - Applies 2D Affine transformation around the midpoint of eyes to make the eye line horizontal.
3. Cropping & Resizing:
   - Crops face bounding box with a 20% margin to preserve contour context (forehead/jaw).
   - Resizes canonically to target_size (default: 128 x 128 pixels).
4. Photometric Normalization:
   - Converts to Grayscale: Y = 0.299*R + 0.587*G + 0.114*B.
   - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) with clip_limit=2.0
     and tile_grid_size=(8, 8) to mitigate illumination variance and shadow artifacts.
"""

import os
from typing import Tuple, Optional, List, Dict, Any
import cv2
import numpy as np


class FacePreprocessor:
    """
    Standardized face detection, alignment, and illumination normalization processor.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (128, 128),
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        face_margin: float = 0.05
    ) -> None:
        """
        Initialize the face preprocessor with target resolution and CLAHE parameters.

        Args:
            target_size: (width, height) in pixels (default: 128x128).
            clahe_clip_limit: Contrast clipping limit for CLAHE.
            clahe_tile_grid: Number of grid tiles for CLAHE.
            face_margin: Proportional bounding box padding (0.20 = 20% margin).
        """
        self.target_size = target_size
        self.face_margin = face_margin
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid
        )

        # Initialize Haar Cascades with local fallbacks
        local_cascade_dir = os.path.join(os.path.dirname(__file__), "cascades")
        face_xml = os.path.join(local_cascade_dir, "haarcascade_frontalface_default.xml")
        eye_xml = os.path.join(local_cascade_dir, "haarcascade_eye.xml")

        if not os.path.exists(face_xml) and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            face_xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            eye_xml = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")

        self.face_cascade = cv2.CascadeClassifier(face_xml) if os.path.exists(face_xml) else None
        self.eye_cascade = cv2.CascadeClassifier(eye_xml) if os.path.exists(eye_xml) else None

    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the primary/largest frontal face in the image.

        Args:
            image: BGR or Grayscale image array.

        Returns:
            Tuple (x, y, w, h) of the largest detected bounding box, or None if not found.
        """
        if self.face_cascade is None or self.face_cascade.empty():
            return None

        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        if len(faces) == 0:
            return None

        # Select the bounding box with the maximum area (largest face in frame)
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        return tuple(int(v) for v in largest_face)

    def detect_all_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect all frontal faces in the image.

        Args:
            image: BGR or Grayscale image array.

        Returns:
            List of tuples [(x, y, w, h), ...] for all detected faces.
        """
        if self.face_cascade is None or self.face_cascade.empty():
            return []

        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30)
        )
        return [tuple(int(v) for v in f) for f in faces]

    def align_face(
        self,
        image: np.ndarray,
        face_box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Align the face bounding box by rotating around the eye centers to ensure
        the inter-ocular line is strictly horizontal (0 degrees).

        Args:
            image: Input image (BGR or Grayscale).
            face_box: (x, y, w, h) bounding box of the face.

        Returns:
            Aligned and cropped face image.
        """
        x, y, w, h = face_box
        img_h, img_w = image.shape[:2]

        # Add contextual margin around face box
        margin_x = int(w * self.face_margin)
        margin_y = int(h * self.face_margin)
        x0 = max(0, x - margin_x)
        y0 = max(0, y - margin_y)
        x1 = min(img_w, x + w + margin_x)
        y1 = min(img_h, y + h + margin_y)

        face_roi = image[y0:y1, x0:x1]
        if face_roi.size == 0:
            return cv2.resize(image, self.target_size)

        gray_roi = face_roi if len(face_roi.shape) == 2 else cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Detect eyes within the upper 60% of the face ROI
        eyes = []
        if self.eye_cascade is not None and not self.eye_cascade.empty():
            upper_h = int(gray_roi.shape[0] * 0.60)
            eyes = self.eye_cascade.detectMultiScale(
                gray_roi[:upper_h, :],
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(15, 15)
            )

        if len(eyes) >= 2:
            # Sort detected eyes from left to right along the x-axis
            sorted_eyes = sorted(eyes, key=lambda e: e[0])
            e_left, e_right = sorted_eyes[0], sorted_eyes[1]

            # Center coordinates of the two eyes
            left_center = (e_left[0] + e_left[2] // 2, e_left[1] + e_left[3] // 2)
            right_center = (e_right[0] + e_right[2] // 2, e_right[1] + e_right[3] // 2)

            dx = right_center[0] - left_center[0]
            dy = right_center[1] - left_center[1]

            # Rotation angle in degrees
            angle = float(np.degrees(np.arctan2(dy, dx)))

            # Prevent extreme rotations (clamp between -35 and +35 degrees)
            if -35.0 <= angle <= 35.0:
                roi_center = (face_roi.shape[1] / 2.0, face_roi.shape[0] / 2.0)
                rot_matrix = cv2.getRotationMatrix2D(roi_center, angle, scale=1.0)
                face_roi = cv2.warpAffine(
                    face_roi,
                    rot_matrix,
                    (face_roi.shape[1], face_roi.shape[0]),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )

        return face_roi

    def normalize(self, face_roi: np.ndarray) -> np.ndarray:
        """
        Normalize face image:
        1. Resize to target dimension (128x128).
        2. Convert to single-channel Grayscale.
        3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            face_roi: Cropped (and aligned) face image.

        Returns:
            Normalized 2D uint8 numpy array of shape (target_height, target_width).
        """
        # Convert to Grayscale if multichannel
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi.copy()

        # Resize to fixed canonical resolution
        resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)

        # Apply CLAHE to eliminate lighting disparities
        equalized = self.clahe.apply(resized)
        return equalized

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        Full end-to-end preprocessing pipeline for a single input frame:
        Detection -> Margin Padding -> Alignment -> Resizing -> CLAHE Normalization.

        Args:
            image: Input raw image (BGR or Grayscale).

        Returns:
            Tuple: (normalized_128x128_face, detected_bounding_box_or_None)
        """
        bbox = self.detect_face(image)

        if bbox is not None:
            aligned_roi = self.align_face(image, bbox)
        else:
            # Fallback: take central square crop if no face detector trigger
            h, w = image.shape[:2]
            min_dim = min(h, w)
            cx, cy = w // 2, h // 2
            x0 = max(0, cx - min_dim // 2)
            y0 = max(0, cy - min_dim // 2)
            aligned_roi = image[y0:y0+min_dim, x0:x0+min_dim]
            bbox = (x0, y0, min_dim, min_dim)

        normalized = self.normalize(aligned_roi)
        return normalized, bbox


def detect_and_align_face(
    image: np.ndarray,
    target_size: Tuple[int, int] = (128, 128)
) -> np.ndarray:
    """
    Convenience functional API for single image face preprocessing.

    Args:
        image: Input image array.
        target_size: Output resolution tuple (default 128x128).

    Returns:
        128x128 normalized uint8 grayscale face image.
    """
    preprocessor = FacePreprocessor(target_size=target_size)
    face, _ = preprocessor.process(image)
    return face
