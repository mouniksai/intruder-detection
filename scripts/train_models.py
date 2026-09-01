"""
train_models.py
===============
Model Training, Validation, Testing, and Persistence Script for Classical ML Pipelines.

Workflow:
1. Loads face dataset (from data/raw, datasets/, or benchmark generator).
2. Performs stratified 3-way partitioning:
   - Training Set (70%): Used to teach models and learn feature representations.
   - Validation Set (15%): Used during model development for hyperparameter checks.
   - Testing Set (15%): Completely unseen data reserved strictly for final testing & confusion matrices.
3. Trains all 5 independent classical ML pipelines.
4. Generates and displays Confusion Matrices for both Training and Testing sets for each of the 5 models.
5. Saves trained pipelines, test samples, and class metadata to `models/review1_models.joblib`
   for instant image classification via `scripts/classify_image.py`.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import joblib

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.preprocessing.dataset_loader import DatasetManager
from src.classifiers.train_eval import Review1Evaluator
from src.classifiers.pipelines import PIPELINE_CONFIGS
from src.open_set.intruder_detector import IntruderDetector


def train_and_save_models(
    raw_data_dir: str = "data/raw",
    model_save_path: str = "models/review1_models.joblib",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42
) -> Dict[str, Any]:
    print("=" * 85)
    print("      CLASSICAL ML INTRUDER DETECTION: TRAINING, VALIDATION & TESTING PIPELINE")
    print("=" * 85)

    dataset_mgr = DatasetManager(raw_data_dir=raw_data_dir)
    print(f"\n[Step 1/5] Loading face dataset from: {raw_data_dir}...")
    X, y, class_names = dataset_mgr.load_from_directory()

    if len(X) == 0:
        print("  -> 'data/raw/' is empty. Bootstrapping with synthetic 5-identity benchmark dataset...")
        X, y, class_names = dataset_mgr.generate_benchmark_dataset(
            num_known_identities=5,
            samples_per_identity=80,
            num_intruder_samples=80,
            random_seed=random_seed
        )

    # Identify enrolled authorized identities (excluding 'unknown', 'intruder', etc.)
    unknown_tags = {"unknown", "intruder", "imposter", "other"}
    enrolled_classes = [c for c in class_names if c.lower() not in unknown_tags]
    all_classes = sorted(list(set(y)))

    # Step 2: Stratified 3-way Split into Train (70%), Val (15%), Test (15%)
    print("\n[Step 2/5] Partitioning dataset into Training, Validation, and Testing sets...")
    from sklearn.model_selection import train_test_split

    # First split off test set (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_seed
    )

    # Next split remaining into train and validation
    adj_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=adj_val_size, stratify=y_train_val, random_state=random_seed
    )

    print("  ---------------------------------------------------------------------------")
    print(f"  - Training Set:   {len(X_train):>4} faces ({(len(X_train)/len(X))*100:.1f}%) -> Used to fit models and learn representations")
    print(f"  - Validation Set: {len(X_val):>4} faces ({(len(X_val)/len(X))*100:.1f}%) -> Used to tune parameters & monitor overfitting")
    print(f"  - Testing Set:    {len(X_test):>4} faces ({(len(X_test)/len(X))*100:.1f}%) -> Strictly unseen data for final evaluation")
    print(f"  - Total Dataset:  {len(X):>4} faces across {len(all_classes)} distinct classes: {all_classes}")
    print("  ---------------------------------------------------------------------------")

    # Step 3: Train and Evaluate all 5 Independent Classical ML Pipelines
    print("\n[Step 3/5] Training & Evaluating 5 Independent Classical ML Pipelines...")
    evaluator = Review1Evaluator()
    summary_df, detailed_results = evaluator.run_all_benchmarks(
        X_train_img=X_train,
        y_train=y_train,
        X_test_img=X_test,
        y_test=y_test,
        class_names=all_classes,
        X_val_img=X_val,
        y_val=y_val
    )

    print("\n" + "=" * 95)
    print("                         MODEL PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 95)
    print(summary_df.to_string(index=False))
    print("=" * 95)

    # Step 4: Display Confusion Matrices for Training and Testing for each Model
    print("\n[Step 4/5] Confusion Matrices across Training and Testing for each of the 5 Models:")
    print("=" * 95)
    for p_id in range(1, 6):
        res = detailed_results[p_id]
        print(f"\n--- {res['pipeline_name']} ---")
        print(f"  Train Accuracy: {res['train_accuracy']*100:.2f}% | Val Accuracy: {res['val_accuracy']*100:.2f}% | Test Accuracy: {res['test_accuracy']*100:.2f}%")
        
        # Test Confusion Matrix DataFrame
        test_cm_df = pd.DataFrame(res["test_confusion_matrix"], index=[f"True_{c}" for c in all_classes], columns=[f"Pred_{c}" for c in all_classes])
        print("\n  [TEST CONFUSION MATRIX (Unseen Test Set)]:")
        print(test_cm_df.to_string())

    # Step 5: Compute Centroid Templates and Gating Thresholds for Enrolled Identities
    print("\n\n[Step 5/5] Computing feature templates and distance thresholds for enrolled identities...")
    enrolled_templates: Dict[str, Dict[int, np.ndarray]] = {}
    template_thresholds: Dict[str, Dict[int, float]] = {}

    for person in enrolled_classes:
        enrolled_templates[person] = {}
        template_thresholds[person] = {}
        person_mask = (y_train == person)

        if not np.any(person_mask):
            continue

        for p_id in range(1, 6):
            if "train" in evaluator.feature_caches and p_id in evaluator.feature_caches["train"]:
                person_feats = evaluator.feature_caches["train"][p_id][person_mask]
            else:
                extractor = PIPELINE_CONFIGS[p_id]["extractor_func"]
                person_imgs = X_train[person_mask]
                person_feats = np.array([extractor(img) for img in person_imgs], dtype=np.float32)

            norms = np.linalg.norm(person_feats, axis=1, keepdims=True) + 1e-7
            person_feats_norm = person_feats / norms

            centroid = np.mean(person_feats_norm, axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-7)

            cos_dists = 1.0 - np.dot(person_feats_norm, centroid)
            max_d = float(np.max(cos_dists)) if len(cos_dists) > 0 else 0.25
            threshold = float(min(max(max_d * 1.5, 0.20), 0.50))

            enrolled_templates[person][p_id] = centroid
            template_thresholds[person][p_id] = threshold

    # Save complete bundle to disk
    os.makedirs(os.path.dirname(os.path.join(WORKSPACE_ROOT, model_save_path)), exist_ok=True)
    full_save_path = os.path.join(WORKSPACE_ROOT, model_save_path)

    saved_payload = {
        "trained_pipelines": evaluator.trained_models,
        "known_classes": enrolled_classes,
        "all_classes": all_classes,
        "enrolled_templates": enrolled_templates,
        "template_thresholds": template_thresholds,
        "prob_threshold": 0.35,
        "knn_dist_threshold": 80.0,
        "min_consensus_votes": 2,
        "summary_df": summary_df,
        "detailed_results": detailed_results,
        # Save test pool for random testing
        "test_images": X_test,
        "test_labels": y_test,
        "val_images": X_val,
        "val_labels": y_val,
        "train_images": X_train,
        "train_labels": y_train
    }

    joblib.dump(saved_payload, full_save_path)
    print(f"\n  -> Successfully serialized trained models & test datasets to: {full_save_path}")
    print(f"\n[Ready] Run 'python scripts/classify_image.py --random' or 'python scripts/classify_image.py <image_path>' to classify images!")

    return saved_payload


if __name__ == "__main__":
    train_and_save_models()
