"""
run_review1_benchmarks.py
=========================
Comprehensive Benchmark and Evaluation Script for Classical ML Pipelines.

Workflow:
1. Discovers and loads face datasets (from `data/raw`, `datasets/`, or synthetic benchmark generator).
2. Performs unified face detection, horizontal eye alignment, and CLAHE normalization.
3. Partitions data with strict methodological differentiation:
   - Training Set (70%): Used to teach models and learn feature representations.
   - Validation Set (15%): Used during development for hyperparameter checks and monitoring generalization.
   - Testing Set (15%): Completely unseen data reserved strictly for final testing & confusion matrices.
   - Held-Out Intruder Pool: Unseen stranger faces for open-set rejection rate benchmarking.
4. Executes all 5 Independent Classical ML Pipelines:
   - Pipeline 1: BSIF + Random Forest
   - Pipeline 2: LPQ + k-NN
   - Pipeline 3: WLD + Logistic Regression
   - Pipeline 4: Gabor Wavelets + RBF-SVM
   - Pipeline 5: Landmark Geometry + Decision Tree
5. Computes multi-class metrics (Accuracy, Precision, Recall, F1-Score, Latency) + Open-Set Intruder Rejection Rates.
6. Generates Markdown summary tables and Confusion Matrix heatmap plots saved in `reports/figures/`.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure root workspace is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.preprocessing.dataset_loader import DatasetManager
from src.classifiers.train_eval import Review1Evaluator
from src.open_set.intruder_detector import IntruderDetector


def main() -> None:
    print("=" * 90)
    print("      CLASSICAL CV & ML INTRUDER DETECTION: BENCHMARK & EVALUATION HARNESS")
    print("=" * 90)

    reports_dir = os.path.join(WORKSPACE_ROOT, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Dataset Loading
    dataset_mgr = DatasetManager()
    print("\n[Step 1/5] Loading & Preprocessing Face Dataset...")

    # Attempt to load from data/raw, else generate benchmark dataset
    X, y, class_names = dataset_mgr.load_from_directory(os.path.join(WORKSPACE_ROOT, "data", "raw"))

    if len(X) == 0:
        print("  -> No local data found in 'data/raw/'. Generating comprehensive 5-identity benchmark dataset...")
        X, y, class_names = dataset_mgr.generate_benchmark_dataset(
            num_known_identities=5,
            samples_per_identity=80,
            num_intruder_samples=100,
            random_seed=42
        )
        print(f"  -> Generated {len(X)} total samples across classes: {class_names}")

    # 2. Stratified Partitioning into Train (70%), Val (15%), Test (15%), and Intruder Pool
    print("\n[Step 2/5] Stratified Partitioning into Training, Validation, Testing, and Intruder Pool...")
    splits = dataset_mgr.split_known_and_unknowns(
        X, y, unknown_labels=["unknown", "intruder"], test_size=0.15, val_size=0.15, random_state=42
    )

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    X_unknown = splits["X_unknown"]
    known_classes = splits["known_classes"]

    print("  ---------------------------------------------------------------------------")
    print(f"  - Training Set:       {len(X_train):>4} faces -> Used for model fitting & feature extraction")
    print(f"  - Validation Set:     {len(X_val):>4} faces -> Used for tuning & monitoring validation performance")
    print(f"  - Testing Set:        {len(X_test):>4} faces -> Completely unseen data for final test confusion matrix")
    print(f"  - Unknown Intruder Pool: {len(X_unknown):>4} faces -> Isolated strangers for open-set security benchmarking")
    print(f"  - Enrolled Identities:   {known_classes}")
    print("  ---------------------------------------------------------------------------")

    # 3. 1-to-N Pipeline Training & Evaluation (5 Features x 5 Classifiers = 25 Experiments)
    print("\n[Step 3/5] Training and Evaluating 1-to-N Classical ML Experiments (5 Features x 5 Classifiers = 25 Experiments)...")
    evaluator = Review1Evaluator(output_dir=figures_dir)
    summary_df, detailed_results = evaluator.run_all_benchmarks(
        X_train_img=X_train,
        y_train=y_train,
        X_test_img=X_test,
        y_test=y_test,
        class_names=known_classes,
        X_val_img=X_val,
        y_val=y_val
    )

    # Programmatic Verification of 25 Experiments
    tuple_keys = [k for k in detailed_results.keys() if isinstance(k, tuple)]
    number_of_experiments = len(tuple_keys)
    unique_features = len(set(k[0] for k in tuple_keys))
    unique_classifiers = len(set(k[1] for k in tuple_keys))

    assert number_of_experiments == 25, f"Expected exactly 25 experiments, got {number_of_experiments}"
    assert unique_features == 5, f"Expected 5 features, got {unique_features}"
    assert unique_classifiers == 5, f"Expected 5 classifiers, got {unique_classifiers}"

    # Display comparison table
    print("\n" + "=" * 105)
    print("                25 FEATURE-CLASSIFIER EXPERIMENTS PERFORMANCE COMPARISON TABLE (5x5)")
    print("=" * 105)
    print(summary_df.to_string(index=False))
    print("=" * 105)

    # 4. Open-Set Intruder Detection Evaluation
    print("\n[Step 4/5] Evaluating Open-Set Intruder Detection & Consensus Fusion Engine...")
    intruder_detector = IntruderDetector(
        trained_pipelines=evaluator.trained_models,
        known_classes=known_classes,
        prob_threshold=0.55,
        knn_dist_threshold=1.20,
        min_consensus_votes=3
    )

    open_set_metrics = intruder_detector.evaluate_open_set_performance(
        X_known_test=X_test,
        y_known_test=y_test,
        X_intruders=X_unknown
    )

    print("\n  --- OPEN-SET RECOGNITION METRICS ---")
    print(f"  - Enrolled Faces Tested:           {open_set_metrics['known_test_samples']}")
    print(f"  - Intruder Faces Tested:           {open_set_metrics['intruder_test_samples']}")
    print(f"  - Correct Identification Rate:     {open_set_metrics['correct_identification_rate_pct']:.2f}%")
    print(f"  - False Rejection Rate (FRR):      {open_set_metrics['false_reject_rate_pct']:.2f}%")
    print(f"  - True Intruder Detection Rate:    {open_set_metrics['intruder_detection_rate_pct']:.2f}%")
    print(f"  - False Acceptance Rate (FAR):     {open_set_metrics['false_accept_rate_pct']:.2f}%")

    # 5. Plot Confusion Matrices & 5x5 Experiment Matrix
    print("\n[Step 5/5] Generating Visual Confusion Matrices and Report Artifacts...")

    # Plot 1: 5x5 Grid of Test Confusion Matrices across all 25 Experiments
    features_list = ["BSIF", "LPQ", "WLD", "Gabor", "Geometry"]
    classifiers_list = ["Random Forest", "KNN", "Logistic Regression", "SVM", "Decision Tree"]

    fig, axes = plt.subplots(5, 5, figsize=(25, 25))
    for r_idx, f_name in enumerate(features_list):
        for c_idx, clf_name in enumerate(classifiers_list):
            res = detailed_results[(f_name, clf_name)]
            cm = res["test_confusion_matrix"]
            ax = axes[r_idx, c_idx]
            sns.heatmap(
                cm,
                annot=False,
                cmap="Blues",
                xticklabels=False,
                yticklabels=False,
                ax=ax,
                cbar=False
            )
            ax.set_title(f"{f_name} + {clf_name}\nAcc: {res['test_accuracy']*100:.1f}%", fontsize=9, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(f"{f_name}\nTrue ID", fontsize=9, fontweight="bold")
            if r_idx == 4:
                ax.set_xlabel(f"{clf_name}\nPred ID", fontsize=9, fontweight="bold")

    plt.suptitle("Complete 5x5 Experiment Grid: 25 Feature-Classifier Test Confusion Matrices", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    cm_test_path = os.path.join(figures_dir, "confusion_matrices_test.png")
    cm_review1_path = os.path.join(figures_dir, "review1_confusion_matrices.png")
    plt.savefig(cm_test_path, dpi=180, bbox_inches='tight')
    plt.savefig(cm_review1_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved 25-Experiment Test Confusion Matrices Plot to: {cm_test_path}")

    # Plot 2: 5x5 Heatmap of Test Accuracy Matrix across Features and Classifiers
    acc_matrix = np.zeros((5, 5), dtype=np.float32)
    for r_idx, f_name in enumerate(features_list):
        for c_idx, clf_name in enumerate(classifiers_list):
            acc_matrix[r_idx, c_idx] = detailed_results[(f_name, clf_name)]["test_accuracy"] * 100.0

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        acc_matrix,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=classifiers_list,
        yticklabels=features_list,
        cbar_kws={'label': 'Test Accuracy (%)'}
    )
    plt.title("25 Experiments Test Accuracy Matrix (5 Features x 5 Classifiers)", fontsize=14, fontweight="bold")
    plt.xlabel("Classifier", fontsize=12, fontweight="bold")
    plt.ylabel("Feature Extractor", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cm_matrix_path = os.path.join(figures_dir, "benchmark_25_matrix.png")
    plt.savefig(cm_matrix_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved 5x5 Test Accuracy Matrix Heatmap to: {cm_matrix_path}")

    # Plot 3: Canonical Train vs Test Comparison
    canonical_pairs = [
        ("BSIF", "Random Forest"),
        ("LPQ", "KNN"),
        ("WLD", "Logistic Regression"),
        ("Gabor", "SVM"),
        ("Geometry", "Decision Tree")
    ]
    fig_comp, axes_comp = plt.subplots(5, 2, figsize=(14, 22))
    for idx, (f_name, clf_name) in enumerate(canonical_pairs):
        res = detailed_results[(f_name, clf_name)]
        train_cm = res["train_confusion_matrix"]
        test_cm = res["test_confusion_matrix"]

        sns.heatmap(
            train_cm,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=known_classes,
            yticklabels=known_classes,
            ax=axes_comp[idx, 0],
            cbar=False
        )
        axes_comp[idx, 0].set_title(f"{f_name} + {clf_name} — [TRAINING SET]\nAccuracy: {res['train_accuracy']*100:.1f}%", fontsize=10, fontweight="bold")
        axes_comp[idx, 0].set_xlabel("Predicted", fontsize=8)
        axes_comp[idx, 0].set_ylabel("True", fontsize=8)

        sns.heatmap(
            test_cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=known_classes,
            yticklabels=known_classes,
            ax=axes_comp[idx, 1],
            cbar=False
        )
        axes_comp[idx, 1].set_title(f"{f_name} + {clf_name} — [TESTING SET (UNSEEN)]\nAccuracy: {res['test_accuracy']*100:.1f}%", fontsize=10, fontweight="bold")
        axes_comp[idx, 1].set_xlabel("Predicted", fontsize=8)
        axes_comp[idx, 1].set_ylabel("True", fontsize=8)

    plt.tight_layout()
    cm_comp_path = os.path.join(figures_dir, "confusion_matrices_train_vs_test.png")
    plt.savefig(cm_comp_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved Train vs Test Confusion Matrices Plot to: {cm_comp_path}")

    # Save Markdown Summary
    md_table_path = os.path.join(reports_dir, "BENCHMARK_RESULTS.md")
    with open(md_table_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark Results & Metrics Summary\n\n")
        f.write("### 25 Classical CV & ML Experiments (5 Features x 5 Classifiers)\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n### Open-Set Intruder Recognition Performance\n\n")
        f.write(f"- **Correct Identification Rate (CIR)**: {open_set_metrics['correct_identification_rate_pct']:.2f}%\n")
        f.write(f"- **False Rejection Rate (FRR)**: {open_set_metrics['false_reject_rate_pct']:.2f}%\n")
        f.write(f"- **True Intruder Detection Rate**: {open_set_metrics['intruder_detection_rate_pct']:.2f}%\n")
        f.write(f"- **False Acceptance Rate (FAR)**: {open_set_metrics['false_accept_rate_pct']:.2f}%\n")
        f.write("\n\n### Confusion Matrix & Heatmap Artifacts\n\n")
        f.write("- 25 Experiments Test Confusion Matrix Grid: `reports/figures/confusion_matrices_test.png`\n")
        f.write("- 5x5 Accuracy Matrix Heatmap: `reports/figures/benchmark_25_matrix.png`\n")
        f.write("- Train vs Test Comparative Confusion Matrices: `reports/figures/confusion_matrices_train_vs_test.png`\n")
    print(f"  -> Saved Benchmark Markdown Summary to: {md_table_path}")

    print("\n" + "=" * 90)
    print("  BENCHMARK COMPLETE: All 25 experiments trained, validated, and tested successfully!")
    print("=" * 90)

    # Required summary printout
    print(f"\nTotal experiments: {number_of_experiments}")
    print(f"Features: {unique_features}")
    print(f"Classifiers: {unique_classifiers}")


if __name__ == "__main__":
    main()

