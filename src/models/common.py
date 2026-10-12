"""
src/models/common.py
====================
Shared utility modules for Review 2 Deep Learning Architectures:
- DeepFaceDataset: PyTorch Dataset with preprocessing & augmentation
- OpenSetGatingEngine: Prototype-based Cosine Distance Rejection Gate
- Training & Evaluation routines
- Benchmark profiling (parameter footprint, model size, inference latency)
- Visualization utilities (confusion matrix and ROC curves)
"""

from typing import Tuple, List, Dict, Optional, Union, Any
import os
import time
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class DeepFaceDataset(Dataset):
    """
    PyTorch Dataset wrapper for face arrays or file paths.
    Converts 128x128 grayscale/BGR faces to 3-channel normalized tensors.
    """

    def __init__(
        self,
        images: Union[np.ndarray, List[np.ndarray], List[str]],
        labels: Optional[Union[np.ndarray, List[Union[int, str]]]] = None,
        target_size: Tuple[int, int] = (128, 128),
        is_training: bool = False,
        augment: bool = False
    ) -> None:
        self.images = images
        self.labels = labels
        self.target_size = target_size
        self.is_training = is_training
        self.augment = augment

        # Base normalization matching ImageNet statistics
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Training augmentation pipeline
        if augment and is_training:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            self.aug_transform = None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[int, str]]:
        item = self.images[idx]
        if isinstance(item, str):
            img = cv2.imread(item)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {item}")
        else:
            img = item.copy()

        # Handle grayscale -> 3 channel conversion
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to model target resolution
        if (img.shape[0], img.shape[1]) != self.target_size:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

        # To PIL / Tensor format [C, H, W] in [0.0, 1.0]
        tensor_img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.aug_transform is not None:
            tensor_img = self.aug_transform(tensor_img)

        tensor_img = self.normalize(tensor_img)

        label = self.labels[idx] if self.labels is not None else 0
        return tensor_img, label


class OpenSetGatingEngine:
    """
    Open-Set Intruder Gating Pipeline based on L2-Normalized Feature Embeddings
    and Centroid Prototype Cosine Distance.

    Workflow:
    1. Feature Embedding: e = f(x) / ||f(x)||_2
    2. Prototype Centroids: c_k = Mean_{x in S_k}(e) / ||Mean||_2
    3. Distance Gating: d* = min_k (1 - cos(e, c_k))
    4. Decision: If d* < tau -> AUTHORIZED (Class = argmin)
                 If d* >= tau -> INTRUDER ALERT (Unknown / Imposter)
    """

    def __init__(self, threshold: float = 0.40) -> None:
        self.threshold = threshold
        self.class_names: List[str] = []
        self.prototypes: Optional[np.ndarray] = None  # Shape: (K, D)

    def fit_prototypes(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        class_names: List[str]
    ) -> None:
        """
        Compute class prototype centroids from normalized training embeddings.
        """
        self.class_names = [c for c in class_names if c not in ["unknown", "intruder"]]
        dim = embeddings.shape[1]
        prototypes = []

        for c_idx, c_name in enumerate(self.class_names):
            mask = (labels == c_name) | (labels == c_idx)
            c_embeddings = embeddings[mask]
            if len(c_embeddings) == 0:
                proto = np.zeros(dim, dtype=np.float32)
            else:
                proto = np.mean(c_embeddings, axis=0)
                norm = np.linalg.norm(proto)
                if norm > 1e-7:
                    proto = proto / norm
            prototypes.append(proto)

        self.prototypes = np.array(prototypes, dtype=np.float32)

    def predict_with_distance(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict identities and minimum cosine distances for input embeddings.

        Returns:
            pred_classes: Array of predicted class strings or 'unknown'
            min_distances: Array of minimum prototype cosine distances in [0, 2]
            best_known_classes: Closest enrolled identity ignoring threshold
        """
        if self.prototypes is None:
            raise ValueError("Engine not fitted with class prototypes.")

        # Cosine similarity: (N, D) @ (K, D).T -> (N, K)
        cosine_sim = np.dot(embeddings, self.prototypes.T)
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        cosine_dist = 1.0 - cosine_sim  # In [0.0, 2.0]

        min_indices = np.argmin(cosine_dist, axis=1)
        min_distances = np.min(cosine_dist, axis=1)

        best_known_classes = np.array([self.class_names[idx] for idx in min_indices])
        pred_classes = np.where(min_distances >= self.threshold, "unknown", best_known_classes)

        return pred_classes, min_distances, best_known_classes

    def evaluate_open_set(
        self,
        known_embeddings: np.ndarray,
        known_labels: np.ndarray,
        unknown_embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate full open-set security metrics across the threshold continuum.
        Computes ROC curve, AUC, FAR vs FRR, and Equal Error Rate (EER).
        """
        # Distances for known authorized samples
        _, known_dists, _ = self.predict_with_distance(known_embeddings)
        # Distances for unseen intruder samples
        _, unknown_dists, _ = self.predict_with_distance(unknown_embeddings)

        # Ground truth: 0 for known (authorized), 1 for intruder (unknown)
        y_true = np.concatenate([np.zeros(len(known_dists)), np.ones(len(unknown_dists))])
        y_scores = np.concatenate([known_dists, unknown_dists])

        # ROC Curve: y_scores are distances. Higher distance -> more likely intruder
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # FAR and FRR calculations across threshold sweep
        taus = np.linspace(0.0, 1.2, 200)
        far_list = []  # False Accept Rate: intruder distance < tau
        frr_list = []  # False Reject Rate: known distance >= tau

        for t in taus:
            far = np.mean(unknown_dists < t) if len(unknown_dists) > 0 else 0.0
            frr = np.mean(known_dists >= t) if len(known_dists) > 0 else 0.0
            far_list.append(far)
            frr_list.append(frr)

        far_arr = np.array(far_list)
        frr_arr = np.array(frr_list)

        # Find Equal Error Rate (EER) where |FAR - FRR| is minimized
        eer_idx = np.argmin(np.abs(far_arr - frr_arr))
        eer = (far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0
        optimal_tau = taus[eer_idx]

        # Calibrate operating threshold to optimal_tau (Equal Error Rate operating point)
        self.threshold = optimal_tau
        pred_known, dist_known, _ = self.predict_with_distance(known_embeddings)
        pred_unknown, dist_unknown, _ = self.predict_with_distance(unknown_embeddings)

        calibrated_far = np.mean(dist_unknown < self.threshold) if len(unknown_dists) > 0 else 0.0
        calibrated_frr = np.mean(dist_known >= self.threshold) if len(known_dists) > 0 else 0.0
        intruder_rejection_rate = 1.0 - calibrated_far

        return {
            "roc_auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "taus": taus,
            "far_curve": far_arr,
            "frr_curve": frr_arr,
            "eer": eer,
            "optimal_tau": optimal_tau,
            "current_tau": self.threshold,
            "current_far": calibrated_far,
            "current_frr": calibrated_frr,
            "intruder_rejection_rate": intruder_rejection_rate,
            "known_dists": known_dists,
            "unknown_dists": unknown_dists
        }


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a PyTorch module."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def compute_model_size_mb(model: nn.Module) -> float:
    """Compute FP32 parameter memory footprint in Megabytes."""
    total_params, _ = count_parameters(model)
    return (total_params * 4.0) / (1024.0 * 1024.0)


def measure_inference_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int] = (3, 128, 128),
    device: torch.device = torch.device("cpu"),
    num_warmup: int = 10,
    num_runs: int = 50
) -> float:
    """Measure average per-frame forward pass latency in milliseconds."""
    model.eval()
    model.to(device)
    dummy_input = torch.randn(1, *input_size, device=device)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = model(dummy_input)
        total_time = time.perf_counter() - start_time

    latency_ms = (total_time / num_runs) * 1000.0
    return latency_ms


def extract_embeddings(
    model: nn.Module,
    dataset: DeepFaceDataset,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu")
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract L2-normalized embeddings from a dataset using the model."""
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch_imgs, batch_lbls in loader:
            batch_imgs = batch_imgs.to(device)
            if hasattr(model, "extract_features"):
                embeds = model.extract_features(batch_imgs)
            else:
                embeds = model(batch_imgs)
            # Ensure L2 normalization
            embeds = F.normalize(embeds, p=2, dim=1)
            all_embeddings.append(embeds.cpu().numpy())
            all_labels.extend(batch_lbls if isinstance(batch_lbls, list) else batch_lbls.numpy() if hasattr(batch_lbls, 'numpy') else list(batch_lbls))

    embeddings_np = np.vstack(all_embeddings)
    labels_np = np.array(all_labels)
    return embeddings_np, labels_np


def train_model(
    model: nn.Module,
    train_dataset: DeepFaceDataset,
    val_dataset: Optional[DeepFaceDataset] = None,
    num_epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True
) -> Dict[str, List[float]]:
    """Train a deep neural network classifier using cross-entropy loss."""
    model.train()
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == lbls).sum().item()
            total += imgs.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        val_loss, val_acc = 0.0, 0.0
        if val_loader:
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    logits = model(imgs)
                    loss = criterion(logits, lbls)
                    val_running_loss += loss.item() * imgs.size(0)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == lbls).sum().item()
                    val_total += imgs.size(0)
            val_loss = val_running_loss / val_total
            val_acc = val_correct / val_total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == num_epochs):
            val_str = f" | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%" if val_loader else ""
            print(f"  Epoch [{epoch:02d}/{num_epochs:02d}] Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}%{val_str}")

    return history


def evaluate_model(
    model: nn.Module,
    test_dataset: DeepFaceDataset,
    class_names: List[str],
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """Evaluate closed-set multi-class classification performance."""
    model.eval()
    model.to(device)
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(lbls.numpy() if hasattr(lbls, 'numpy') else list(lbls))

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "class_names": class_names
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    output_path: str
) -> None:
    """Save formatted confusion matrix heatmap plot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Identity", fontsize=10)
    plt.ylabel("True Identity", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_open_set_roc(
    open_set_results: Dict[str, Any],
    model_name: str,
    output_path: str
) -> None:
    """Save ROC curve and FAR vs FRR threshold sweep plot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: ROC Curve
    fpr = open_set_results["fpr"]
    tpr = open_set_results["tpr"]
    roc_auc = open_set_results["roc_auc"]
    eer = open_set_results["eer"]

    ax1.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance")
    ax1.scatter([eer], [1.0 - eer], color="red", s=80, zorder=5, label=f"EER Point ({eer:.2%})")
    ax1.set_title(f"{model_name}: Open-Set ROC Curve", fontsize=11, fontweight="bold")
    ax1.set_xlabel("False Accept Rate (FAR)", fontsize=10)
    ax1.set_ylabel("True Accept Rate (1 - FRR)", fontsize=10)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    # Subplot 2: FAR vs FRR vs Threshold
    taus = open_set_results["taus"]
    far_curve = open_set_results["far_curve"]
    frr_curve = open_set_results["frr_curve"]
    opt_tau = open_set_results["optimal_tau"]

    ax2.plot(taus, far_curve, color="#d62728", lw=2, label="FAR (Imposter Accepted)")
    ax2.plot(taus, frr_curve, color="#2ca02c", lw=2, label="FRR (Authorized Rejected)")
    ax2.axvline(opt_tau, color="purple", linestyle="--", lw=1.5, label=f"Optimal tau* = {opt_tau:.2f}")
    ax2.set_title(f"{model_name}: Error Rates vs. Distance Threshold", fontsize=11, fontweight="bold")
    ax2.set_xlabel(r"Cosine Distance Threshold $\tau$", fontsize=10)
    ax2.set_ylabel("Error Rate", fontsize=10)
    ax2.set_xlim([0.0, 1.2])
    ax2.set_ylim([-0.02, 1.02])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="center right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
