"""
train_eval.py
=============
Training, Validation, Testing, and Benchmarking Harness for Classical ML Pipelines.

Strict Protocol Differentiation:
- Training: Trains the 5 classical feature extractors & classifiers.
- Validation: Tunes hyperparameters, evaluates intermediate generalization, monitors overfitting.
- Testing: Completely isolated test set evaluated strictly once to report final generalization & confusion matrices.

Computes:
- Train, Validation, and Test Accuracies, Precisions, Recalls, and F1-Scores
- Per-class precision, recall, and f1 breakdowns
- Confusion Matrices on Test (and Train) data
- Feature extraction latency and inference throughput
"""

from typing import Dict, List, Tuple, Any, Optional
import time
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, List, Tuple, Any, Optional, Union
import time
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from .pipelines import (
    FEATURE_EXTRACTORS,
    CLASSIFIER_FACTORIES,
    PIPELINE_CONFIGS
)


class Review1Evaluator:
    """
    Evaluator harness for executing 1-to-N classical ML experiments:
    5 Feature Extractors x 5 Classifiers = 25 Experiments across Train, Validation, and Test datasets.
    """

    def __init__(self, output_dir: str = "reports/figures") -> None:
        """
        Initialize the Evaluator.

        Args:
            output_dir: Directory where figures and confusion matrix plots are saved.
        """
        self.output_dir = output_dir
        self.trained_models: Dict[Union[int, Tuple[str, str]], Any] = {}
        self.feature_caches: Dict[str, Dict[str, np.ndarray]] = {}

    def extract_features_for_dataset(
        self,
        X_images: np.ndarray,
        feature_key: Union[str, int],
        cache_key: Optional[str] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Extract features for all images in X using the specified feature extractor.
        Caches feature representations to avoid redundant recomputations across classifiers.

        Args:
            X_images: Array of preprocessed face images shape (N, 128, 128).
            feature_key: Feature name (e.g. 'BSIF') or index (1 to 5).
            cache_key: Optional cache identifier (e.g. 'train', 'val', 'test').

        Returns:
            X_feat: Feature matrix shape (N, D).
            avg_latency_ms: Average extraction latency per face in milliseconds.
        """
        feat_name = feature_key
        if isinstance(feature_key, int):
            feat_name = PIPELINE_CONFIGS.get(feature_key, {}).get("feature_name", str(feature_key))

        # Check cache
        if cache_key and cache_key in self.feature_caches and feat_name in self.feature_caches[cache_key]:
            return self.feature_caches[cache_key][feat_name], 0.0

        if feat_name in FEATURE_EXTRACTORS:
            extractor_func = FEATURE_EXTRACTORS[feat_name]["extractor_func"]
        elif isinstance(feature_key, int) and feature_key in PIPELINE_CONFIGS:
            extractor_func = PIPELINE_CONFIGS[feature_key]["extractor_func"]
        else:
            raise KeyError(f"Unknown feature extractor: {feature_key}")

        num_samples = len(X_images)
        if num_samples == 0:
            return np.empty((0, 0), dtype=np.float32), 0.0

        feat_list = []
        t0 = time.perf_counter()
        for i in range(num_samples):
            feat = extractor_func(X_images[i])
            feat_list.append(feat)
        t_total = time.perf_counter() - t0

        X_feat = np.array(feat_list, dtype=np.float32)
        avg_latency_ms = (t_total / num_samples) * 1000.0

        if cache_key:
            if cache_key not in self.feature_caches:
                self.feature_caches[cache_key] = {}
            self.feature_caches[cache_key][feat_name] = X_feat
            if isinstance(feature_key, int):
                self.feature_caches[cache_key][feature_key] = X_feat

        return X_feat, avg_latency_ms

    def train_and_evaluate_experiment(
        self,
        feature_name: str,
        classifier_name: str,
        X_train_img: np.ndarray,
        y_train: np.ndarray,
        X_test_img: np.ndarray,
        y_test: np.ndarray,
        class_names: List[str],
        X_val_img: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train a specific (Feature, Classifier) pair and evaluate across Training, Validation, and Testing sets.

        Args:
            feature_name: Name of feature extractor (e.g. 'BSIF', 'LPQ', 'WLD', 'Gabor', 'Geometry').
            classifier_name: Name of classifier (e.g. 'Random Forest', 'KNN', 'Logistic Regression', 'SVM', 'Decision Tree').
            X_train_img: Training images (N_train, 128, 128).
            y_train: Training labels (N_train,).
            X_test_img: Testing images (N_test, 128, 128).
            y_test: Testing labels (N_test,).
            class_names: List of unique class labels.
            X_val_img: Optional validation images (N_val, 128, 128).
            y_val: Optional validation labels (N_val,).

        Returns:
            Dictionary containing metrics, confusion matrices, timing, and trained model.
        """
        if feature_name not in FEATURE_EXTRACTORS:
            raise KeyError(f"Invalid feature extractor: {feature_name}")
        if classifier_name not in CLASSIFIER_FACTORIES:
            raise KeyError(f"Invalid classifier: {classifier_name}")

        clf_info = CLASSIFIER_FACTORIES[classifier_name]
        pipeline_factory = clf_info["factory"]

        # 1. Feature Extraction (Reuses cached feature representation)
        X_train_feat, train_ext_lat = self.extract_features_for_dataset(X_train_img, feature_name, "train")
        X_test_feat, test_ext_lat = self.extract_features_for_dataset(X_test_img, feature_name, "test")

        feat_dim = X_train_feat.shape[1]

        # 2. Classifier Pipeline Training on Training Set
        pipeline = pipeline_factory()

        t_train_start = time.perf_counter()
        pipeline.fit(X_train_feat, y_train)
        t_train_sec = time.perf_counter() - t_train_start

        # 3. Train Set Evaluation
        y_train_pred = pipeline.predict(X_train_feat)
        train_acc = float(accuracy_score(y_train, y_train_pred))
        train_cm = confusion_matrix(y_train, y_train_pred, labels=class_names)

        # 4. Optional Validation Set Evaluation
        val_acc = None
        val_prec = None
        val_rec = None
        val_f1 = None
        val_cm = None
        if X_val_img is not None and len(X_val_img) > 0 and y_val is not None and len(y_val) > 0:
            X_val_feat, _ = self.extract_features_for_dataset(X_val_img, feature_name, "val")
            y_val_pred = pipeline.predict(X_val_feat)
            val_acc = float(accuracy_score(y_val, y_val_pred))
            val_prec = float(precision_score(y_val, y_val_pred, average='macro', zero_division=0))
            val_rec = float(recall_score(y_val, y_val_pred, average='macro', zero_division=0))
            val_f1 = float(f1_score(y_val, y_val_pred, average='macro', zero_division=0))
            val_cm = confusion_matrix(y_val, y_val_pred, labels=class_names)

        # 5. Test Set Evaluation & Latency
        t_infer_start = time.perf_counter()
        y_test_pred = pipeline.predict(X_test_feat)
        t_infer_total = time.perf_counter() - t_infer_start
        avg_infer_lat_ms = (t_infer_total / len(y_test)) * 1000.0 if len(y_test) > 0 else 0.0

        # Save trained pipeline under (feature, classifier) tuple key
        self.trained_models[(feature_name, classifier_name)] = pipeline

        # 6. Test Metrics & Confusion Matrix
        test_acc = float(accuracy_score(y_test, y_test_pred))
        test_prec = float(precision_score(y_test, y_test_pred, average='macro', zero_division=0))
        test_rec = float(recall_score(y_test, y_test_pred, average='macro', zero_division=0))
        test_f1 = float(f1_score(y_test, y_test_pred, average='macro', zero_division=0))
        test_cm = confusion_matrix(y_test, y_test_pred, labels=class_names)
        clf_report = classification_report(y_test, y_test_pred, labels=class_names, zero_division=0, output_dict=True)

        exp_name = f"{feature_name} -> {classifier_name}"

        return {
            "experiment_key": (feature_name, classifier_name),
            "experiment_name": exp_name,
            "feature_name": feature_name,
            "classifier_name": classifier_name,
            "feature_dim": feat_dim,
            # Training metrics
            "train_accuracy": train_acc,
            "train_time_sec": t_train_sec,
            "train_confusion_matrix": train_cm,
            # Validation metrics
            "val_accuracy": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1_score": val_f1,
            "val_confusion_matrix": val_cm,
            # Testing metrics
            "accuracy": test_acc,
            "test_accuracy": test_acc,
            "precision": test_prec,
            "test_precision": test_prec,
            "recall": test_rec,
            "test_recall": test_rec,
            "f1_score": test_f1,
            "test_f1_score": test_f1,
            "extract_latency_ms": test_ext_lat,
            "infer_latency_ms": avg_infer_lat_ms,
            "total_latency_ms": test_ext_lat + avg_infer_lat_ms,
            "confusion_matrix": test_cm,
            "test_confusion_matrix": test_cm,
            "classification_report": clf_report,
            "y_test": y_test,
            "y_pred": y_test_pred,
            "pipeline": pipeline
        }

    def train_and_evaluate_pipeline(
        self,
        pipeline_id: int,
        X_train_img: np.ndarray,
        y_train: np.ndarray,
        X_test_img: np.ndarray,
        y_test: np.ndarray,
        class_names: List[str],
        X_val_img: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Backward-compatible single pipeline evaluation method (mapped to canonical pair).
        """
        config = PIPELINE_CONFIGS[pipeline_id]
        res = self.train_and_evaluate_experiment(
            feature_name=config["feature_name"],
            classifier_name=config["classifier_name"],
            X_train_img=X_train_img,
            y_train=y_train,
            X_test_img=X_test_img,
            y_test=y_test,
            class_names=class_names,
            X_val_img=X_val_img,
            y_val=y_val
        )
        res["pipeline_id"] = pipeline_id
        res["model_id"] = pipeline_id
        res["pipeline_name"] = config["pipeline_name"]
        res["model_name"] = config["model_name"]
        self.trained_models[pipeline_id] = res["pipeline"]
        return res

    train_and_evaluate_member = train_and_evaluate_pipeline

    def run_all_benchmarks(
        self,
        X_train_img: np.ndarray,
        y_train: np.ndarray,
        X_test_img: np.ndarray,
        y_test: np.ndarray,
        class_names: List[str],
        X_val_img: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], Dict[str, Any]]]:
        """
        Execute 1-to-N benchmarking: 5 Feature Extractors x 5 Classifiers = 25 Experiments.
        Reuses extracted feature representations across classifiers.

        Returns:
            summary_df: Formatted pandas DataFrame containing all 25 experiment results.
            detailed_results: Raw results dictionary keyed by (feature_name, classifier_name).
        """
        detailed_results: Dict[Any, Dict[str, Any]] = {}
        table_rows = []

        has_val = X_val_img is not None and len(X_val_img) > 0

        # Canonical mapping for backward compatibility
        canonical_pairs = {
            1: ("BSIF", "Random Forest"),
            2: ("LPQ", "KNN"),
            3: ("WLD", "Logistic Regression"),
            4: ("Gabor", "SVM"),
            5: ("Geometry", "Decision Tree")
        }

        # 1-to-N Nested Loop Structure: 5 Features x 5 Classifiers = 25 Experiments
        for feat_name, feat_info in FEATURE_EXTRACTORS.items():
            for clf_name, clf_info in CLASSIFIER_FACTORIES.items():
                exp_key = (feat_name, clf_name)
                res = self.train_and_evaluate_experiment(
                    feature_name=feat_name,
                    classifier_name=clf_name,
                    X_train_img=X_train_img,
                    y_train=y_train,
                    X_test_img=X_test_img,
                    y_test=y_test,
                    class_names=class_names,
                    X_val_img=X_val_img,
                    y_val=y_val
                )
                detailed_results[exp_key] = res

                # Map to canonical pipeline IDs 1..5 for backward compatibility if matched
                for p_id, pair in canonical_pairs.items():
                    if pair == exp_key:
                        detailed_results[p_id] = res
                        self.trained_models[p_id] = res["pipeline"]

                row = {
                    "Feature Extractor": feat_name,
                    "Classifier": clf_name,
                    "Feature Dim": res["feature_dim"],
                    "Train Acc (%)": f"{res['train_accuracy'] * 100:.1f}%",
                }
                if has_val and res["val_accuracy"] is not None:
                    row["Val Acc (%)"] = f"{res['val_accuracy'] * 100:.1f}%"
                row.update({
                    "Test Acc (%)": f"{res['test_accuracy'] * 100:.1f}%",
                    "Precision": f"{res['test_precision']:.4f}",
                    "Recall": f"{res['test_recall']:.4f}",
                    "F1-Score": f"{res['test_f1_score']:.4f}",
                    "Latency (ms/face)": f"{res['total_latency_ms']:.2f} ms"
                })
                table_rows.append(row)

        summary_df = pd.DataFrame(table_rows)
        return summary_df, detailed_results

