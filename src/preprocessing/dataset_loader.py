"""
dataset_loader.py
=================
Dataset management, stratified splitting, and augmentation for Open-Set Intruder Recognition.

Handles:
1. Loading raw images from directory structures: `data/raw/<class_name>/*.jpg` or `datasets/`.
2. Parsing bounding box CSVs if present (e.g. `datasets/1/faces.csv` or `datasets/2/selfie_id.csv`).
3. Running unified preprocessing via FacePreprocessor.
4. Stratified splitting into Enrolled Known (Train / Validation / Test) and Held-Out Unknowns / Intruders.
5. Providing a self-contained synthetic/benchmark face generator with controlled illumination,
   orientation, noise, and identity-specific geometric/texture signatures for instant reproducible testing.
"""

from typing import Tuple, List, Dict, Optional, Union
import os
import glob
from pathlib import Path
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from .detect_align import FacePreprocessor


def safe_imread(path: str) -> Optional[np.ndarray]:
    """Safely read an image file handling Windows unicode/accented paths."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if len(data) == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(path)


class DatasetManager:
    """
    Manages loading, preprocessing, partitioning, and augmenting dataset images for Review 1.
    """

    def __init__(
        self,
        raw_data_dir: str = "data/raw",
        processed_data_dir: str = "data/processed",
        preprocessor: Optional[FacePreprocessor] = None
    ) -> None:
        """
        Initialize the Dataset Manager.

        Args:
            raw_data_dir: Path to directory containing raw images grouped by person subfolder.
            processed_data_dir: Path to directory for cached aligned/normalized face images.
            preprocessor: FacePreprocessor instance (creates default if None).
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.preprocessor = preprocessor or FacePreprocessor(target_size=(128, 128))

    def load_from_directory(
        self,
        base_dir: Optional[str] = None,
        include_datasets_folder: bool = True,
        max_samples_per_person: int = 40
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load images from subdirectories where each subdirectory name is a class label.
        Also automatically searches and ingests all real human identities from the `datasets/` folder.

        Args:
            base_dir: Root directory containing person folders.
            include_datasets_folder: Whether to automatically ingest subjects from `datasets/`.
            max_samples_per_person: Maximum number of face images to take per identity for balanced training.

        Returns:
            X: Array of preprocessed face images, shape (N, 128, 128).
            y: Array of string class labels, shape (N,).
            class_names: List of unique class names.
        """
        search_dir = base_dir or self.raw_data_dir
        images_list: List[np.ndarray] = []
        labels_list: List[str] = []

        import unicodedata

        def clean_ascii(text: str) -> str:
            normalized = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
            return normalized.replace(' ', '_').strip()

        # Cache file path
        cache_file = os.path.join(self.processed_data_dir, "face_preprocessed_cache.npz")
        if os.path.exists(cache_file):
            print(f"  [DatasetManager] Loading preprocessed faces from fast cache: {cache_file}...")
            cached = np.load(cache_file, allow_pickle=True)
            X = cached["X"]
            y = cached["y"]
            class_names = sorted(list(set(y)))
            print(f"  [DatasetManager] Loaded {len(X)} cached faces across {len(class_names)} unique identities.")
            return X, y, class_names

        # 1. Load from data/raw/
        if os.path.exists(search_dir):
            subdirs = [
                d for d in os.listdir(search_dir)
                if os.path.isdir(os.path.join(search_dir, d))
            ]
            for raw_label in sorted(subdirs):
                label = clean_ascii(raw_label)
                folder_path = os.path.join(search_dir, raw_label)
                pattern_jpg = os.path.join(folder_path, "*.[jJ][pP][gG]")
                pattern_png = os.path.join(folder_path, "*.[pP][nN][gG]")
                pattern_jpeg = os.path.join(folder_path, "*.[jJ][pP][eE][gG]")
                image_paths = (glob.glob(pattern_jpg) + glob.glob(pattern_png) + glob.glob(pattern_jpeg))[:max_samples_per_person]

                for img_path in image_paths:
                    img = safe_imread(img_path)
                    if img is None:
                        continue
                    face, _ = self.preprocessor.process(img)
                    images_list.append(face)
                    labels_list.append(label)

        # 2. Automatically ingest real identities from `datasets/`
        if include_datasets_folder:
            datasets_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
            if os.path.exists(datasets_root):
                d2_folders = glob.glob(os.path.join(datasets_root, "2", "Selfies ID Images dataset", "*", "*"))
                for folder in d2_folders:
                    if not os.path.isdir(folder):
                        continue
                    folder_name = os.path.basename(folder)
                    if "_name_" in folder_name:
                        raw_person_name = folder_name.split("_name_")[-1].strip()
                    else:
                        raw_person_name = folder_name
                    person_name = clean_ascii(raw_person_name)

                    img_paths = (glob.glob(os.path.join(folder, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(folder, "*.[pP][nN][gG]")))[:max_samples_per_person]
                    for p in img_paths:
                        img = safe_imread(p)
                        if img is None:
                            continue
                        face, _ = self.preprocessor.process(img)
                        images_list.append(face)
                        labels_list.append(person_name)

        X = np.array(images_list, dtype=np.uint8)
        y = np.array(labels_list, dtype=str)
        class_names = sorted(list(set(labels_list)))

        # Cache preprocessed faces for instant loading on subsequent runs
        os.makedirs(self.processed_data_dir, exist_ok=True)
        np.savez_compressed(cache_file, X=X, y=y)
        print(f"  [DatasetManager] Processed & cached {len(X)} total real human face images across {len(class_names)} unique identities to {cache_file}.")
        return X, y, class_names

    def split_known_and_unknowns(
        self,
        X: np.ndarray,
        y: np.ndarray,
        unknown_labels: Optional[List[str]] = None,
        test_size: float = 0.20,
        val_size: float = 0.10,
        random_state: int = 42
    ) -> Dict[str, Union[np.ndarray, List[str]]]:
        """
        Partition dataset into Known Train/Val/Test and an isolated Unknown/Intruder Test Pool.

        Args:
            X: Array of images shape (N, 128, 128).
            y: Array of string labels shape (N,).
            unknown_labels: Labels considered intruders/unknowns (default: ['unknown', 'intruder']).
            test_size: Test fraction for known classes.
            val_size: Validation fraction for known classes.
            random_state: Random seed for reproducibility.

        Returns:
            Dictionary with 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test',
            'X_unknown', 'y_unknown', 'known_classes'.
        """
        if unknown_labels is None:
            unknown_labels = ["unknown", "intruder", "imposter", "other"]

        # Convert unknown labels to lowercase for comparison
        unknown_set = set(lbl.lower() for lbl in unknown_labels)

        is_unknown = np.array([lbl.lower() in unknown_set for lbl in y])
        is_known = ~is_unknown

        X_known = X[is_known]
        y_known = y[is_known]

        X_unknown = X[is_unknown]
        y_unknown = y[is_unknown]

        known_classes = sorted(list(set(y_known)))

        if len(X_known) == 0:
            raise ValueError("No known enrolled samples found in dataset.")

        # Stratified train/test split on known classes
        X_train, X_test, y_train, y_test = train_test_split(
            X_known,
            y_known,
            test_size=test_size,
            stratify=y_known,
            random_state=random_state
        )

        # Optional validation split from training set
        if val_size > 0.0:
            adj_val_size = val_size / (1.0 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=adj_val_size,
                stratify=y_train,
                random_state=random_state
            )
        else:
            X_val, y_val = np.empty((0, 128, 128), dtype=np.uint8), np.empty((0,), dtype=str)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "X_unknown": X_unknown,
            "y_unknown": y_unknown,
            "known_classes": known_classes
        }

    @staticmethod
    def generate_benchmark_dataset(
        num_known_identities: int = 5,
        samples_per_identity: int = 80,
        num_intruder_samples: int = 100,
        image_size: Tuple[int, int] = (128, 128),
        random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate a fully synthetic benchmark dataset with parameterized facial structures,
        micro-textures, lighting variations, and geometric configurations.

        Useful for instant end-to-end unit testing and algorithm verification.

        Args:
            num_known_identities: Number of enrolled team members (default: 5).
            samples_per_identity: Number of image samples per enrolled member.
            num_intruder_samples: Number of held-out intruder face samples.
            image_size: (H, W) resolution (default: 128x128).
            random_seed: Reproducibility seed.

        Returns:
            X: Array of images shape (Total_Samples, 128, 128).
            y: Array of class labels shape (Total_Samples,).
            class_names: List of distinct class names.
        """
        np.random.seed(random_seed)
        H, W = image_size

        known_names = [f"Subject_{i+1}" for i in range(num_known_identities)]
        all_images: List[np.ndarray] = []
        all_labels: List[str] = []

        # Generate unique facial parameters for each known identity
        identity_profiles: Dict[str, Dict[str, Any]] = {}
        for name in known_names:
            identity_profiles[name] = {
                "base_tone": np.random.uniform(110, 180),
                "eye_dist": np.random.uniform(28, 42),
                "eye_y": np.random.uniform(45, 55),
                "nose_w": np.random.uniform(10, 20),
                "nose_len": np.random.uniform(18, 30),
                "mouth_w": np.random.uniform(24, 40),
                "mouth_y": np.random.uniform(88, 100),
                "texture_freq": np.random.uniform(0.05, 0.25),
                "texture_angle": np.random.uniform(0, np.pi),
            }

        # Helper function to synthesize a single stylized face
        def synthesize_face(params: Dict[str, Any], is_intruder: bool = False) -> np.ndarray:
            canvas = np.zeros((H, W), dtype=np.float32)

            # 1. Base Head Ellipse
            center = (W // 2 + int(np.random.normal(0, 1.5)), H // 2 + int(np.random.normal(0, 1.5)))
            axes = (int(W * 0.38 + np.random.normal(0, 1)), int(H * 0.44 + np.random.normal(0, 1)))
            cv2.ellipse(canvas, center, axes, 0, 0, 360, params["base_tone"], -1)

            # 2. Add local texture pattern (simulating micro skin/hair texture)
            xx, yy = np.meshgrid(np.arange(W), np.arange(H))
            rot_coord = xx * np.cos(params["texture_angle"]) + yy * np.sin(params["texture_angle"])
            texture = np.sin(2 * np.pi * params["texture_freq"] * rot_coord) * 12.0
            canvas = np.clip(canvas + texture * (canvas > 10), 0, 255)

            # 3. Eyes with eyebrows
            eye_y = int(params["eye_y"] + np.random.normal(0, 0.8))
            eye_spacing = params["eye_dist"] / 2.0
            left_eye_x = int(center[0] - eye_spacing)
            right_eye_x = int(center[0] + eye_spacing)

            # Eyeballs & Pupils
            cv2.circle(canvas, (left_eye_x, eye_y), 5, 40.0, -1)
            cv2.circle(canvas, (right_eye_x, eye_y), 5, 40.0, -1)
            cv2.circle(canvas, (left_eye_x, eye_y), 2, 220.0, -1)
            cv2.circle(canvas, (right_eye_x, eye_y), 2, 220.0, -1)

            # Eyebrows
            cv2.line(canvas, (left_eye_x - 7, eye_y - 6), (left_eye_x + 7, eye_y - 7), 30.0, 2)
            cv2.line(canvas, (right_eye_x - 7, eye_y - 7), (right_eye_x + 7, eye_y - 6), 30.0, 2)

            # 4. Nose Bridge & Nostrils
            nose_top_y = eye_y + 4
            nose_bot_y = int(nose_top_y + params["nose_len"])
            cv2.line(canvas, (center[0], nose_top_y), (center[0], nose_bot_y), 70.0, 2)
            nw = int(params["nose_w"] / 2.0)
            cv2.circle(canvas, (center[0] - nw, nose_bot_y), 2, 50.0, -1)
            cv2.circle(canvas, (center[0] + nw, nose_bot_y), 2, 50.0, -1)

            # 5. Mouth & Lips
            mouth_y = int(params["mouth_y"] + np.random.normal(0, 1.0))
            mw = int(params["mouth_w"] / 2.0)
            cv2.ellipse(canvas, (center[0], mouth_y), (mw, 4), 0, 0, 360, 60.0, -1)
            cv2.line(canvas, (center[0] - mw, mouth_y), (center[0] + mw, mouth_y), 30.0, 1)

            # 6. Lighting gradient & sensor noise
            lighting_slope = np.random.uniform(-0.3, 0.3)
            light_grad = np.outer(np.ones(H), np.linspace(-1, 1, W)) * (lighting_slope * 40.0)
            canvas = np.clip(canvas + light_grad, 0, 255)

            noise = np.random.normal(0, 3.0, (H, W))
            canvas = np.clip(canvas + noise, 0, 255).astype(np.uint8)

            # Run through standard preprocessor CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(canvas)

        # Generate known enrolled samples
        for name in known_names:
            profile = identity_profiles[name]
            for _ in range(samples_per_identity):
                # Add minor jitter to person parameters for within-class intra-variance
                jittered = profile.copy()
                jittered["base_tone"] += np.random.normal(0, 3.0)
                jittered["texture_angle"] += np.random.normal(0, 0.05)
                face = synthesize_face(jittered, is_intruder=False)
                all_images.append(face)
                all_labels.append(name)

        # Generate unknown intruder samples
        for _ in range(num_intruder_samples):
            # Completely random facial parameters for unknown imposters
            random_profile = {
                "base_tone": np.random.uniform(90, 200),
                "eye_dist": np.random.uniform(22, 48),
                "eye_y": np.random.uniform(40, 60),
                "nose_w": np.random.uniform(8, 25),
                "nose_len": np.random.uniform(15, 35),
                "mouth_w": np.random.uniform(20, 48),
                "mouth_y": np.random.uniform(82, 105),
                "texture_freq": np.random.uniform(0.02, 0.40),
                "texture_angle": np.random.uniform(0, np.pi),
            }
            face = synthesize_face(random_profile, is_intruder=True)
            all_images.append(face)
            all_labels.append("unknown")

        X = np.array(all_images, dtype=np.uint8)
        y = np.array(all_labels, dtype=str)
        class_names = known_names + ["unknown"]
        return X, y, class_names
