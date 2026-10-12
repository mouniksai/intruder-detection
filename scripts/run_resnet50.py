"""
scripts/run_resnet50.py
=======================
Independent Training, Evaluation, and Open-Set Intruder Testing for ResNet-50 (2015).

Workflow:
1. Load dataset (data/raw or synthetic benchmark).
2. Stratified split: Train (70%), Val (15%), Test (15%), and Intruder Pool (unknowns).
3. Initialize ResNet50Face architecture.
4. Profile parameter count and model memory size.
5. Train model with CrossEntropy loss and monitoring on validation set.
6. Evaluate closed-set test metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
7. Fit prototype centroids on L2-normalized feature embeddings.
8. Evaluate open-set intruder detection:
   - Sweeps distance threshold tau to compute FAR, FRR, ROC-AUC, and Equal Error Rate (EER).
   - Measures imposter rejection rate at optimal threshold.
9. Measure inference latency (ms/frame).
10. Save confusion matrix and ROC curve plots in reports/figures/.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.preprocessing.dataset_loader import DatasetManager
from src.models.resnet50 import ResNet50Face
from src.models.common import (
    DeepFaceDataset,
    OpenSetGatingEngine,
    train_model,
    evaluate_model,
    extract_embeddings,
    count_parameters,
    compute_model_size_mb,
    measure_inference_latency,
    plot_confusion_matrix,
    plot_open_set_roc
)


def run_resnet50(
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 6e-4,
    device_name: str = "cpu",
    output_dir: str = "reports/figures"
):
    print("=" * 80)
    print("  ARCHITECTURE 4: ResNet-50 (2015) — Residual Identity Highways (ArcFace Backbone)")
    print("=" * 80)

    device = torch.device(device_name if torch.cuda.is_available() and device_name != "cpu" else "cpu")
    print(f"[*] Execution Device: {device}")

    # 1. Dataset Loading
    mgr = DatasetManager()
    data_path = os.path.join(WORKSPACE_ROOT, "data", "raw")
    X, y, class_names = mgr.load_from_directory(data_path)
    if len(X) == 0:
        print("[*] Generating 5-identity face benchmark dataset with held-out intruders...")
        X, y, class_names = mgr.generate_benchmark_dataset(
            num_known_identities=5,
            samples_per_identity=80,
            num_intruder_samples=100,
            image_size=(128, 128),
            random_seed=42
        )

    # 2. Stratified Partitioning
    splits = mgr.split_known_and_unknowns(
        X, y, unknown_labels=["unknown", "intruder"], test_size=0.15, val_size=0.15, random_state=42
    )

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]
    X_unknown = splits["X_unknown"]
    known_classes = splits["known_classes"]
    num_classes = len(known_classes)

    class_to_idx = {name: idx for idx, name in enumerate(known_classes)}
    y_train_idx = np.array([class_to_idx[l] for l in y_train])
    y_val_idx = np.array([class_to_idx[l] for l in y_val])
    y_test_idx = np.array([class_to_idx[l] for l in y_test])

    print(f"[*] Enrolled Classes ({num_classes}): {known_classes}")
    print(f"[*] Train Faces: {len(X_train)} | Val Faces: {len(X_val)} | Test Faces: {len(X_test)} | Unknown Intruders: {len(X_unknown)}")

    # 3. Model Initialization & Architecture Profiling
    model = ResNet50Face(num_classes=num_classes, embedding_dim=512)
    total_params, trainable_params = count_parameters(model)
    model_size_mb = compute_model_size_mb(model)

    print(f"\n[Architecture Specs]")
    print(f"  • Total Layers:        50 (Residual Bottlenecks)")
    print(f"  • Total Parameters:    {total_params:,} ({total_params/1e6:.2f} Million)")
    print(f"  • Trainable Params:    {trainable_params:,}")
    print(f"  • Model Size (FP32):   {model_size_mb:.2f} MB")
    print(f"  • Key Structure:       Bottleneck Residual Blocks + Identity Shortcut Connections")

    # 4. Training
    train_dataset = DeepFaceDataset(X_train, y_train_idx, target_size=(128, 128), is_training=True, augment=True)
    val_dataset = DeepFaceDataset(X_val, y_val_idx, target_size=(128, 128), is_training=False, augment=False)
    test_dataset = DeepFaceDataset(X_test, y_test_idx, target_size=(128, 128), is_training=False, augment=False)
    intruder_dataset = DeepFaceDataset(X_unknown, [0]*len(X_unknown), target_size=(128, 128), is_training=False, augment=False)

    print(f"\n[*] Training ResNet-50 for {epochs} epochs...")
    train_model(model, train_dataset, val_dataset, num_epochs=epochs, batch_size=batch_size, lr=lr, device=device)

    # 5. Closed-Set Testing
    print("\n[*] Evaluating Closed-Set Test Performance...")
    eval_res = evaluate_model(model, test_dataset, known_classes, device=device)
    print(f"  • Test Accuracy:   {eval_res['accuracy']*100:.2f}%")
    print(f"  • Test Precision:  {eval_res['precision']:.4f}")
    print(f"  • Test Recall:     {eval_res['recall']:.4f}")
    print(f"  • Test F1-Score:   {eval_res['f1_score']:.4f}")

    # 6. Feature Embedding Extraction & Prototype Cosine Gating
    print("\n[*] Extracting L2-Normalized Feature Embeddings for Open-Set Gating...")
    train_embed_ds = DeepFaceDataset(X_train, y_train_idx, target_size=(128, 128), is_training=False, augment=False)
    train_embeds, _ = extract_embeddings(model, train_embed_ds, device=device)
    test_embeds, _ = extract_embeddings(model, test_dataset, device=device)
    intruder_embeds, _ = extract_embeddings(model, intruder_dataset, device=device)

    gating_engine = OpenSetGatingEngine(threshold=0.35)
    gating_engine.fit_prototypes(train_embeds, y_train_idx, known_classes)

    open_set_res = gating_engine.evaluate_open_set(test_embeds, y_test_idx, intruder_embeds)
    print(f"  • Open-Set ROC AUC:              {open_set_res['roc_auc']:.4f}")
    print(f"  • Equal Error Rate (EER):        {open_set_res['eer']*100:.2f}%")
    print(f"  • Optimal Distance Threshold:    tau* = {open_set_res['optimal_tau']:.3f}")
    print(f"  • Imposter Rejection Rate:       {open_set_res['intruder_rejection_rate']*100:.2f}%")
    print(f"  • False Accept Rate (FAR):       {open_set_res['current_far']*100:.2f}%")
    print(f"  • False Reject Rate (FRR):       {open_set_res['current_frr']*100:.2f}%")

    # 7. Latency Profiling
    latency_ms = measure_inference_latency(model, input_size=(3, 128, 128), device=device)
    print(f"\n[*] Average Inference Latency: {latency_ms:.2f} ms / frame ({1000.0/latency_ms:.1f} FPS)")

    # 8. Visualizations
    out_dir = os.path.join(WORKSPACE_ROOT, output_dir)
    cm_path = os.path.join(out_dir, "confusion_matrix_resnet50.png")
    roc_path = os.path.join(out_dir, "roc_resnet50.png")
    plot_confusion_matrix(eval_res["confusion_matrix"], known_classes, "ResNet-50 Confusion Matrix", cm_path)
    plot_open_set_roc(open_set_res, "ResNet-50 (2015)", roc_path)
    print(f"[*] Visualizations saved to:\n  - {cm_path}\n  - {roc_path}")

    # 9. Theoretical Performance Justification
    print("\n" + "-" * 80)
    print("  THEORETICAL JUSTIFICATION FOR RESNET-50 PERFORMANCE")
    print("-" * 80)
    print("  • Why it performed this way:")
    print("    1. Identity shortcut connections (y = F(x) + x) create an unobstructed gradient highway,")
    print("       allowing 50 deep layers to converge without vanishing gradients or network degradation.")
    print("    2. Identity shortcuts preserve low-level facial landmark alignments and spatial anchors")
    print("       across deep stages, producing highly compact intra-class clusters and broad inter-class")
    print("       separations. This makes ResNet-50 the universal gold standard backbone for modern face")
    print("       recognition systems (ArcFace, CosFace, InsightFace).")
    print("=" * 80)

    return {
        "architecture": "ResNet-50",
        "parameters_m": total_params / 1e6,
        "model_size_mb": model_size_mb,
        "test_acc": eval_res["accuracy"],
        "precision": eval_res["precision"],
        "recall": eval_res["recall"],
        "f1_score": eval_res["f1_score"],
        "roc_auc": open_set_res["roc_auc"],
        "eer": open_set_res["eer"],
        "optimal_tau": open_set_res["optimal_tau"],
        "rejection_rate": open_set_res["intruder_rejection_rate"],
        "far": open_set_res["current_far"],
        "frr": open_set_res["current_frr"],
        "latency_ms": latency_ms
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ResNet-50 face recognition benchmark")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    run_resnet50(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device_name=args.device)
