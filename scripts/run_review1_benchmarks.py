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

    # 3. Pipeline Training & Evaluation
    print("\n[Step 3/5] Training and Evaluating 5 Independent Classical ML Pipelines...")
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

    # Display comparison table
    print("\n" + "=" * 95)
    print("                   5 INDEPENDENT PIPELINE PERFORMANCE COMPARISON TABLE")
    print("=" * 95)
    print(summary_df.to_string(index=False))
    print("=" * 95)

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

    # 5. Plot Confusion Matrices
    print("\n[Step 5/5] Generating Visual Confusion Matrices and Report Artifacts...")

    # Plot 1: Test Confusion Matrices for all 5 Models
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for idx, p_id in enumerate(range(1, 6)):
        res = detailed_results[p_id]
        cm = res["test_confusion_matrix"]
        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=known_classes,
            yticklabels=known_classes,
            ax=ax,
            cbar=False
        )
        ax.set_title(f"{res['pipeline_name']}\nTest Accuracy: {res['test_accuracy']*100:.1f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Identity", fontsize=9)
        ax.set_ylabel("True Identity", fontsize=9)

    # Hide 6th unused subplot
    axes[5].axis('off')

    plt.tight_layout()
    cm_test_path = os.path.join(figures_dir, "confusion_matrices_test.png")
    cm_review1_path = os.path.join(figures_dir, "review1_confusion_matrices.png")
    plt.savefig(cm_test_path, dpi=200, bbox_inches='tight')
    plt.savefig(cm_review1_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved Test Confusion Matrices Plot to: {cm_test_path}")

    # Plot 2: Side-by-Side Train vs Test Confusion Matrices for each Model
    fig_comp, axes_comp = plt.subplots(5, 2, figsize=(14, 22))
    for idx, p_id in enumerate(range(1, 6)):
        res = detailed_results[p_id]
        train_cm = res["train_confusion_matrix"]
        test_cm = res["test_confusion_matrix"]

        # Train CM
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
        axes_comp[idx, 0].set_title(f"{res['pipeline_name']} — [TRAINING SET]\nAccuracy: {res['train_accuracy']*100:.1f}%", fontsize=10, fontweight="bold")
        axes_comp[idx, 0].set_xlabel("Predicted", fontsize=8)
        axes_comp[idx, 0].set_ylabel("True", fontsize=8)

        # Test CM
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
        axes_comp[idx, 1].set_title(f"{res['pipeline_name']} — [TESTING SET (UNSEEN)]\nAccuracy: {res['test_accuracy']*100:.1f}%", fontsize=10, fontweight="bold")
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
        f.write("### 5 Independent Classical CV & ML Pipeline Comparison\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n### Open-Set Intruder Recognition Performance\n\n")
        f.write(f"- **Correct Identification Rate (CIR)**: {open_set_metrics['correct_identification_rate_pct']:.2f}%\n")
        f.write(f"- **False Rejection Rate (FRR)**: {open_set_metrics['false_reject_rate_pct']:.2f}%\n")
        f.write(f"- **True Intruder Detection Rate**: {open_set_metrics['intruder_detection_rate_pct']:.2f}%\n")
        f.write(f"- **False Acceptance Rate (FAR)**: {open_set_metrics['false_accept_rate_pct']:.2f}%\n")
        f.write("\n\n### Confusion Matrix Artifacts\n\n")
        f.write("- Test Confusion Matrices: `reports/figures/confusion_matrices_test.png`\n")
        f.write("- Train vs Test Comparative Confusion Matrices: `reports/figures/confusion_matrices_train_vs_test.png`\n")
    print(f"  -> Saved Benchmark Markdown Summary to: {md_table_path}")

    print("\n" + "=" * 90)
    print("  BENCHMARK COMPLETE: All 5 pipelines trained, validated, and tested successfully!")
    print("=" * 90)


if __name__ == "__main__":
    main()
