"""
scripts/generate_review2_visuals.py
===================================
Generates high-resolution comparative visualization charts for Review 2 Demo:
1. All 5 ROC Curves combined on a single plot for direct discriminative comparison.
2. Latency vs. Accuracy vs. Model Footprint bubble/scatter plot.
3. Comparative bar chart of Closed-Set Accuracy vs. Imposter Rejection Rate.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = os.path.join(WORKSPACE_ROOT, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Actual benchmarked experimental data from the 5 standalone runs
BENCHMARK_DATA = [
    {
        "name": "AlexNet (2012)",
        "short": "AlexNet",
        "acc": 20.0,
        "f1": 0.067,
        "auc": 0.740,
        "eer": 46.0,
        "rejection": 54.0,  # Calibrated at EER point
        "latency": 29.12,
        "params": 42.32,
        "size_mb": 161.45,
        "color": "#e74c3c",  # Red
        "verdict": "Poor / Baseline"
    },
    {
        "name": "VGG-16 (2014)",
        "short": "VGG-16",
        "acc": 100.0,
        "f1": 1.000,
        "auc": 0.900,
        "eer": 15.67,
        "rejection": 72.0,
        "latency": 57.30,
        "params": 32.55,
        "size_mb": 124.17,
        "color": "#9b59b6",  # Purple
        "verdict": "Good / High Latency"
    },
    {
        "name": "GoogLeNet (2014)",
        "short": "GoogLeNet",
        "acc": 95.0,
        "f1": 0.949,
        "auc": 0.826,
        "eer": 28.67,
        "rejection": 46.0,
        "latency": 19.79,
        "params": 6.79,
        "size_mb": 25.92,
        "color": "#f39c12",  # Orange
        "verdict": "Good / Efficient"
    },
    {
        "name": "ResNet-50 (2015)",
        "short": "ResNet-50",
        "acc": 100.0,
        "f1": 1.000,
        "auc": 0.908,
        "eer": 18.00,
        "rejection": 84.0,
        "latency": 35.54,
        "params": 24.56,
        "size_mb": 93.69,
        "color": "#27ae60",  # Green
        "verdict": "State-of-the-Art"
    },
    {
        "name": "MobileNetV2 (2018)",
        "short": "MobileNetV2",
        "acc": 61.67,
        "f1": 0.506,
        "auc": 0.746,
        "eer": 28.00,
        "rejection": 69.0,
        "latency": 11.96,
        "params": 2.88,
        "size_mb": 11.00,
        "color": "#2980b9",  # Blue
        "verdict": "Optimal Edge Speed"
    }
]


def generate_accuracy_vs_latency_plot():
    fig, ax = plt.subplots(figsize=(10, 6))

    for item in BENCHMARK_DATA:
        # Scatter with size proportional to parameter count
        bubble_size = item["params"] * 25
        ax.scatter(
            item["latency"],
            item["acc"],
            s=bubble_size,
            color=item["color"],
            alpha=0.65,
            edgecolors="black",
            linewidth=1.5,
            label=f"{item['short']} ({item['params']:.1f}M params, {item['latency']:.1f}ms)"
        )
        # Annotation offset
        ax.annotate(
            f"{item['short']}\n({item['acc']:.1f}% Acc / {item['latency']:.1f}ms)",
            (item["latency"], item["acc"]),
            textcoords="offset points",
            xytext=(0, 14 if item["acc"] < 90 else -25),
            ha='center',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec=item["color"])
        )

    ax.set_title("Review 2: Closed-Set Accuracy vs. Inference Latency (Bubble Size = Parameter Footprint)", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Inference Latency (ms per frame) → Lower is Faster", fontsize=11)
    ax.set_ylabel("Closed-Set Test Accuracy (%) → Higher is Better", fontsize=11)
    ax.set_ylim([10, 110])
    ax.set_xlim([5, 65])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    out_path = os.path.join(FIGURES_DIR, "review2_latency_vs_accuracy.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[*] Saved: {out_path}")


def generate_metric_comparison_barchart():
    fig, ax = plt.subplots(figsize=(11, 6))

    labels = [d["short"] for d in BENCHMARK_DATA]
    acc_scores = [d["acc"] for d in BENCHMARK_DATA]
    auc_scores = [d["auc"] * 100 for d in BENCHMARK_DATA]
    rej_scores = [d["rejection"] for d in BENCHMARK_DATA]

    x = np.arange(len(labels))
    width = 0.25

    rects1 = ax.bar(x - width, acc_scores, width, label='Closed-Set Accuracy (%)', color='#2b5c8f', edgecolor='black')
    rects2 = ax.bar(x, auc_scores, width, label='Open-Set ROC AUC (×100)', color='#e67e22', edgecolor='black')
    rects3 = ax.bar(x + width, rej_scores, width, label='Imposter Rejection Rate (%)', color='#27ae60', edgecolor='black')

    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title('Review 2 Deep Learning Benchmark: Accuracy vs. Open-Set Rejection by Architecture', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_ylim([0, 115])
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    ax.legend(loc='upper left', fontsize=10)

    # Add values on top of bars
    for rects in [rects1, rects2, rects3]:
        for bar in rects:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    out_path = os.path.join(FIGURES_DIR, "review2_metric_comparison_bars.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[*] Saved: {out_path}")


if __name__ == "__main__":
    generate_accuracy_vs_latency_plot()
    generate_metric_comparison_barchart()
