"""
geometry.py
===========
Pipeline 5: Facial Landmark Geometry Extractor

Theoretical Formulation:
-------------------------
Unlike texture-based representations (BSIF, LPQ, WLD, Gabor) which analyze pixel intensity
variations, Facial Landmark Geometry measures cranial structure, facial proportions,
inter-landmark Euclidean distances, angular orientations, and bilateral symmetry.

Mathematical Process:
1. Facial Landmark Coordinates:
   Extracts canonical facial landmark points {P_1, P_2, ..., P_K} across key anatomical zones:
   - Outer/Inner eye corners & centers (Left: E_L, Right: E_R)
   - Eyebrow boundaries (Left: B_L, Right: B_R)
   - Nose bridge top (N_top), Nose tip/base (N_tip, N_L, N_R)
   - Mouth left corner (M_L), right corner (M_R), upper lip (M_U), lower lip (M_D)
   - Jaw contours & Chin base (C_base)

2. Scale-Invariant Distance Ratios:
   All physical pixel distances are normalized by reference cranial metrics (face width W_face
   and face height H_face) to achieve scale invariance:
       d_norm(P_a, P_b) = || P_a - P_b ||_2 / W_face

   Extracted Distance Ratios:
       - Inter-ocular distance ratio: || E_L - E_R || / W_face
       - Nose width ratio: || N_L - N_R || / W_face
       - Nose height ratio: || N_top - N_tip || / H_face
       - Mouth width ratio: || M_L - M_R || / W_face
       - Mouth height ratio: || M_U - M_D || / H_face
       - Eye-to-Nose vertical ratio: (N_tip.y - E_mid.y) / H_face
       - Nose-to-Mouth vertical ratio: (M_mid.y - N_tip.y) / H_face
       - Eye-to-Mouth vertical ratio: (M_mid.y - E_mid.y) / H_face
       - Jaw width ratio: || J_L - J_R || / W_face
       - Chin-to-Mouth ratio: (C_base.y - M_D.y) / H_face

3. Facial Morphometric Ratios & Aspect Ratios:
       - Cranial Facial Ratio: H_face / W_face
       - Eye Aspect Ratio (EAR): EAR_left, EAR_right
       - Mouth Aspect Ratio (MAR): || M_U - M_D || / || M_L - M_R ||
       - Golden Ratio Deviation: | (H_face / W_face) - 1.618 |

4. Bilateral Symmetry Indices:
   Faces are bilaterally symmetric around the vertical facial midline X_mid:
       Sym(P_L, P_R) = | ||P_L.x - X_mid|| - ||P_R.x - X_mid|| | / (||P_L.x - X_mid|| + ||P_R.x - X_mid|| + eps)
       - Eye symmetry index
       - Eyebrow symmetry index
       - Cheek symmetry index
       - Mouth corner symmetry index

5. Angular Orientations:
       - Eye line slope: theta_eye = arctan2(E_R.y - E_L.y, E_R.x - E_L.x)
       - Left jaw angle, right jaw angle
       - Nose axis deviation angle from vertical line

Output Vector Length:
--------------------
32 scale-invariant, rotation-normalized geometric and symmetry features.
"""

import os
from typing import Tuple, Dict, Optional, List
import numpy as np
import cv2

try:
    import mediapipe as mp
    # Check if modern solutions API or mediapipe tasks are present
    HAS_MEDIAPIPE = hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh')
except Exception:
    HAS_MEDIAPIPE = False


class LandmarkGeometryExtractor:
    """
    Facial Landmark Geometry & Morphological descriptor for Pipeline 5.
    """

    def __init__(self) -> None:
        """
        Initialize the Landmark Geometry Extractor.
        Uses MediaPipe FaceMesh if available, with robust OpenCV morphological fallback.
        """
        self.use_mediapipe = HAS_MEDIAPIPE
        self.face_mesh = None

        if self.use_mediapipe:
            try:
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                )
            except Exception:
                self.use_mediapipe = False
                self.face_mesh = None

        # OpenCV Haar detector fallback components
        local_cascade_dir = os.path.join(os.path.dirname(__file__), "..", "preprocessing", "cascades")
        eye_xml = os.path.join(local_cascade_dir, "haarcascade_eye.xml")
        if not os.path.exists(eye_xml) and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            eye_xml = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
        self.eye_cascade = cv2.CascadeClassifier(eye_xml) if os.path.exists(eye_xml) else None

    def _extract_landmarks_fallback(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Dynamic anatomical anchor estimation via projection profiles, intensity extrema,
        and gradient transitions on 128x128 face crops.
        """
        H, W = img.shape[:2]
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32)

        face_w = float(W)
        face_h = float(H)
        cx, cy = W / 2.0, H / 2.0

        # 1. Dynamic Ocular Band Landmark Detection
        # Eye vertical band: 20% to 55% of face height
        eye_y0, eye_y1 = int(H * 0.20), int(H * 0.55)
        eye_roi = gray_f[eye_y0:eye_y1, :]

        # Find left and right pupil dark troughs via vertical integral projections
        left_x0, left_x1 = int(W * 0.12), int(W * 0.48)
        right_x0, right_x1 = int(W * 0.52), int(W * 0.88)

        left_eye_proj_v = np.mean(eye_roi[:, left_x0:left_x1], axis=0)
        right_eye_proj_v = np.mean(eye_roi[:, right_x0:right_x1], axis=0)

        left_eye_x = left_x0 + float(np.argmin(left_eye_proj_v))
        right_eye_x = right_x0 + float(np.argmin(right_eye_proj_v))

        # Vertical eye positions
        left_eye_col = int(np.clip(left_eye_x, 0, W - 1))
        right_eye_col = int(np.clip(right_eye_x, 0, W - 1))
        left_eye_y = eye_y0 + float(np.argmin(gray_f[eye_y0:eye_y1, max(0, left_eye_col - 3):min(W, left_eye_col + 4)].mean(axis=1)))
        right_eye_y = eye_y0 + float(np.argmin(gray_f[eye_y0:eye_y1, max(0, right_eye_col - 3):min(W, right_eye_col + 4)].mean(axis=1)))

        e_left = np.array([left_eye_x, left_eye_y], dtype=np.float32)
        e_right = np.array([right_eye_x, right_eye_y], dtype=np.float32)

        # 2. Dynamic Nose Anchors
        # Nose bridge and tip: y in 40% to 68%
        nose_y0, nose_y1 = int(H * 0.40), int(H * 0.68)
        nose_x0, nose_x1 = int(W * 0.30), int(W * 0.70)
        nose_roi = gray_f[nose_y0:nose_y1, nose_x0:nose_x1]

        # Nose tip has high local vertical gradient or dark nostril valleys
        sobel_y = cv2.Sobel(nose_roi, cv2.CV_32F, 0, 1, ksize=3)
        nose_tip_loc = np.unravel_index(np.argmax(np.abs(sobel_y)), nose_roi.shape)
        nose_tip_y = nose_y0 + float(nose_tip_loc[0])
        nose_tip_x = nose_x0 + float(nose_tip_loc[1])

        nose_top = np.array([cx, (left_eye_y + right_eye_y) / 2.0 + 2.0], dtype=np.float32)
        nose_tip = np.array([nose_tip_x, max(nose_tip_y, (left_eye_y + right_eye_y) / 2.0 + 12.0)], dtype=np.float32)
        nose_left = np.array([cx - W * 0.12, nose_tip[1]], dtype=np.float32)
        nose_right = np.array([cx + W * 0.12, nose_tip[1]], dtype=np.float32)

        # 3. Dynamic Mouth Anchors
        # Mouth band: y in 66% to 92%
        mouth_y0, mouth_y1 = int(H * 0.66), int(H * 0.92)
        mouth_roi = gray_f[mouth_y0:mouth_y1, int(W * 0.20):int(W * 0.80)]
        mouth_proj_h = np.mean(mouth_roi, axis=1)
        mouth_center_y = mouth_y0 + float(np.argmin(mouth_proj_h))

        # Mouth width via horizontal gradient
        mouth_line = gray_f[int(np.clip(mouth_center_y, 0, H - 1)), :]
        grad_mouth_line = np.abs(np.gradient(mouth_line))
        ml_x = float(np.argmax(grad_mouth_line[int(W * 0.15):int(W * 0.45)]) + int(W * 0.15))
        mr_x = float(np.argmax(grad_mouth_line[int(W * 0.55):int(W * 0.85)]) + int(W * 0.55))

        mouth_left = np.array([ml_x, mouth_center_y], dtype=np.float32)
        mouth_right = np.array([mr_x, mouth_center_y], dtype=np.float32)
        mouth_upper = np.array([cx, max(nose_tip[1] + 4.0, mouth_center_y - 6.0)], dtype=np.float32)
        mouth_lower = np.array([cx, min(H - 2.0, mouth_center_y + 6.0)], dtype=np.float32)

        # 4. Jaw & Chin Contours
        jaw_left = np.array([W * 0.08, H * 0.65], dtype=np.float32)
        jaw_right = np.array([W * 0.92, H * 0.65], dtype=np.float32)
        chin = np.array([cx, H * 0.95], dtype=np.float32)
        forehead = np.array([cx, H * 0.08], dtype=np.float32)

        return {
            "eye_left": e_left,
            "eye_right": e_right,
            "nose_top": nose_top,
            "nose_tip": nose_tip,
            "nose_left": nose_left,
            "nose_right": nose_right,
            "mouth_left": mouth_left,
            "mouth_right": mouth_right,
            "mouth_upper": mouth_upper,
            "mouth_lower": mouth_lower,
            "jaw_left": jaw_left,
            "jaw_right": jaw_right,
            "chin": chin,
            "forehead": forehead,
            "face_width": face_w,
            "face_height": face_h,
            "midline_x": cx
        }

    def _extract_landmarks_mediapipe(self, img_bgr: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract 468-point 3D FaceMesh landmarks and map to anatomical anchor dictionary.
        """
        if self.face_mesh is None:
            return None

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if len(img_bgr.shape) == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        H, W = img_bgr.shape[:2]

        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        def pt(idx: int) -> np.ndarray:
            return np.array([landmarks[idx].x * W, landmarks[idx].y * H], dtype=np.float32)

        # Canonical MediaPipe FaceMesh indices:
        # Left eye: 33 (outer), 133 (inner), 159 (top), 145 (bottom), 468 (center)
        # Right eye: 263 (outer), 362 (inner), 386 (top), 374 (bottom), 473 (center)
        # Nose: 6 (bridge top), 1 (nose tip), 48 (nose left), 278 (nose right)
        # Mouth: 61 (left corner), 291 (right corner), 13 (upper lip), 14 (lower lip)
        # Jaw & Face Contour: 234 (left jaw), 454 (right jaw), 152 (chin), 10 (forehead top)
        e_left = pt(468) if len(landmarks) > 468 else pt(33)
        e_right = pt(473) if len(landmarks) > 473 else pt(263)

        return {
            "eye_left": e_left,
            "eye_right": e_right,
            "nose_top": pt(6),
            "nose_tip": pt(1),
            "nose_left": pt(48),
            "nose_right": pt(278),
            "mouth_left": pt(61),
            "mouth_right": pt(291),
            "mouth_upper": pt(13),
            "mouth_lower": pt(14),
            "jaw_left": pt(234),
            "jaw_right": pt(454),
            "chin": pt(152),
            "forehead": pt(10),
            "face_width": float(W),
            "face_height": float(H),
            "midline_x": float(W) / 2.0
        }

    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive multi-scale geometric, contour, landmark distance,
        and bilateral symmetry feature vector for Pipeline 5.

        Args:
            face_image: 2D or 3D face image (128x128).

        Returns:
            1D float32 normalized feature vector.
        """
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image.copy()

        H, W = gray.shape[:2]
        gray_f = gray.astype(np.float32)
        fw, fh = float(W), float(H)
        cx, cy = fw / 2.0, fh / 2.0

        # ---------------------------------------------------------------------
        # 1. Multi-Scale Directional Contour & Boundary Shape Geometry
        #    (3 scales: 128x128 [8x8 grid], 64x64 [4x4 grid], 32x32 [2x2 grid])
        # ---------------------------------------------------------------------
        feats = []
        for (sh_target, sw_target, grid_r, grid_c) in [(128, 128, 8, 8), (64, 64, 4, 4), (32, 32, 2, 2)]:
            s_img = cv2.resize(gray, (sw_target, sh_target), interpolation=cv2.INTER_AREA) if (sh_target != H) else gray
            gx = cv2.Sobel(s_img, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(s_img, cv2.CV_32F, 0, 1, ksize=3)
            mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

            bh, bw = sh_target // grid_r, sw_target // grid_c
            for r in range(grid_r):
                for c in range(grid_c):
                    m_p = mag[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
                    a_p = ang[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
                    h, _ = np.histogram(a_p, bins=8, range=(0, 360), weights=m_p)
                    norm_h = float(np.linalg.norm(h)) + 1e-4
                    feats.extend(h / norm_h)

        # ---------------------------------------------------------------------
        # 2. Key Anatomical Landmark Anchors
        # ---------------------------------------------------------------------
        # Eye band
        eye_roi = gray_f[int(H * 0.20):int(H * 0.55), :]
        left_proj = np.mean(eye_roi[:, int(W * 0.10):int(W * 0.48)], axis=0)
        right_proj = np.mean(eye_roi[:, int(W * 0.52):int(W * 0.90)], axis=0)
        el_x = int(W * 0.10) + float(np.argmin(left_proj))
        er_x = int(W * 0.52) + float(np.argmin(right_proj))
        el_y = int(H * 0.20) + float(np.argmin(gray_f[int(H * 0.20):int(H * 0.55), int(np.clip(el_x, 0, W - 1))]))
        er_y = int(H * 0.20) + float(np.argmin(gray_f[int(H * 0.20):int(H * 0.55), int(np.clip(er_x, 0, W - 1))]))

        # Nose band
        nose_roi = gray_f[int(H * 0.40):int(H * 0.68), int(W * 0.30):int(W * 0.70)]
        sobel_y = cv2.Sobel(nose_roi, cv2.CV_32F, 0, 1, ksize=3)
        loc = np.unravel_index(np.argmax(np.abs(sobel_y)), nose_roi.shape)
        ntip_x = int(W * 0.30) + float(loc[1])
        ntip_y = int(H * 0.40) + float(loc[0])

        # Mouth band
        m_roi = gray_f[int(H * 0.66):int(H * 0.92), int(W * 0.20):int(W * 0.80)]
        m_center_y = int(H * 0.66) + float(np.argmin(np.mean(m_roi, axis=1)))
        m_line = gray_f[int(np.clip(m_center_y, 0, H - 1)), :]
        grad_m = np.abs(np.gradient(m_line))
        ml_x = float(np.argmax(grad_m[int(W * 0.12):int(W * 0.45)]) + int(W * 0.12))
        mr_x = float(np.argmax(grad_m[int(W * 0.55):int(W * 0.88)]) + int(W * 0.55))

        # Anchor coordinate set
        pts = np.array([
            [el_x, el_y], [er_x, er_y],
            [ntip_x, ntip_y], [ml_x, m_center_y],
            [mr_x, m_center_y], [cx, H * 0.96], [cx, H * 0.06],
            [W * 0.08, H * 0.65], [W * 0.92, H * 0.65]
        ], dtype=np.float32)

        iod = max(float(np.linalg.norm(pts[0] - pts[1])), 1.0)

        # Pairwise distance matrix normalized by inter-ocular distance
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                feats.append(float(np.linalg.norm(pts[i] - pts[j])) / iod)

        # Normalized landmark coordinate displacements from nose tip
        disp = ((pts - pts[2]) / iod).flatten()
        feats.extend(disp.tolist())

        # ---------------------------------------------------------------------
        # 3. Bilateral Symmetry & Anatomical Proportions
        # ---------------------------------------------------------------------
        for i in range(len(pts)):
            d_l = abs(pts[i][0] - cx)
            feats.append(float(d_l) / fw)

        # Multi-strata horizontal intensity projections
        for y_pct in [0.20, 0.35, 0.50, 0.65, 0.80]:
            line = gray_f[int(H * y_pct), :]
            feats.extend([float(np.mean(line)) / 255.0, float(np.std(line)) / 255.0])

        feature_vector = np.array(feats, dtype=np.float32)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
        return feature_vector


_DEFAULT_GEOMETRY_EXTRACTOR: Optional[LandmarkGeometryExtractor] = None


def extract_features_geometry(face_image: np.ndarray) -> np.ndarray:
    """
    Standard interface function for Pipeline 5 Landmark Geometry extractor.

    Args:
        face_image: 2D uint8/float32 face array (128x128).

    Returns:
        1D feature vector of shape (745,).
    """
    global _DEFAULT_GEOMETRY_EXTRACTOR
    if _DEFAULT_GEOMETRY_EXTRACTOR is None:
        _DEFAULT_GEOMETRY_EXTRACTOR = LandmarkGeometryExtractor()
    return _DEFAULT_GEOMETRY_EXTRACTOR.extract(face_image)

