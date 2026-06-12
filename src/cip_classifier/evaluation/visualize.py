"""Visualization utilities for evaluation results.

Generates confusion matrices, accuracy bar charts, per-field breakdowns,
and confidence histograms. Uses matplotlib (no interactive display needed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .metrics import ClassificationMetrics
from .predictions import PredictionSet


def plot_comparison_bars(
    metrics_list: list[ClassificationMetrics],
    output_path: Path,
    title: str = "Model Comparison",
) -> None:
    """Bar chart comparing major/broad accuracy across models."""
    import matplotlib.pyplot as plt

    models = [m.model_name for m in metrics_list]
    major_acc = [m.major_field_accuracy for m in metrics_list]
    broad_acc = [m.broad_field_accuracy for m in metrics_list]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))
    ax.bar(x - width / 2, major_acc, width, label="Major Field Acc")
    ax.bar(x + width / 2, broad_acc, width, label="Broad Field Acc")

    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.7, color="r", linestyle="--", alpha=0.5, label="Major target (0.7)")
    ax.axhline(y=0.9, color="g", linestyle="--", alpha=0.5, label="Broad target (0.9)")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comparison chart: {output_path}")


def plot_confusion_matrix(
    pred_set: PredictionSet,
    output_path: Path,
    level: str = "broad",
    max_classes: int = 25,
) -> None:
    """Plot a confusion matrix heatmap.

    Args:
        level: "broad" or "major". Major may be too large for readable display.
        max_classes: Cap number of classes shown (top by support).
    """
    import matplotlib.pyplot as plt
    from collections import Counter

    preds = pred_set.predictions
    if level == "broad":
        true_labels = [p.true_broad_field for p in preds]
        pred_labels = [p.predicted_broad_field for p in preds]
    else:
        true_labels = [p.true_major_field for p in preds]
        pred_labels = [p.predicted_major_field for p in preds]

    # Get top classes by support
    label_counts = Counter(true_labels)
    top_labels = [lbl for lbl, _ in label_counts.most_common(max_classes)]
    label_to_idx = {lbl: i for i, lbl in enumerate(top_labels)}

    n = len(top_labels)
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        if t in label_to_idx and p in label_to_idx:
            matrix[label_to_idx[t], label_to_idx[p]] += 1

    fig, ax = plt.subplots(figsize=(max(10, n * 0.5), max(8, n * 0.4)))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")

    # Shorten labels for display
    short_labels = [lbl[:25] for lbl in top_labels]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({level} field) — {pred_set.model_name}")

    plt.colorbar(im)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix: {output_path}")


def plot_confidence_histogram(
    pred_set: PredictionSet,
    output_path: Path,
) -> None:
    """Histogram of confidence scores, colored by correct/incorrect."""
    import matplotlib.pyplot as plt

    confidences_correct = []
    confidences_wrong = []
    for p in pred_set.predictions:
        if p.confidence is not None:
            if p.predicted_major_field == p.true_major_field:
                confidences_correct.append(p.confidence)
            else:
                confidences_wrong.append(p.confidence)

    if not confidences_correct and not confidences_wrong:
        print("No confidence scores available, skipping histogram.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 30)
    if confidences_correct:
        ax.hist(confidences_correct, bins=bins, alpha=0.6, label="Correct", color="green")
    if confidences_wrong:
        ax.hist(confidences_wrong, bins=bins, alpha=0.6, label="Incorrect", color="red")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title(f"Confidence Distribution — {pred_set.model_name}")
    ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved confidence histogram: {output_path}")


def plot_per_field_f1(
    metrics: ClassificationMetrics,
    output_path: Path,
    top_n: int = 30,
) -> None:
    """Horizontal bar chart of per-field F1 scores (worst to best)."""
    import matplotlib.pyplot as plt

    if not metrics.per_field_metrics:
        print("No per-field metrics available.")
        return

    sorted_fields = sorted(
        metrics.per_field_metrics.items(), key=lambda x: x[1]["f1"]
    )[:top_n]

    fields = [f[0][:40] for f in sorted_fields]
    f1_scores = [f[1]["f1"] for f in sorted_fields]

    fig, ax = plt.subplots(figsize=(10, max(6, len(fields) * 0.3)))
    y_pos = range(len(fields))
    ax.barh(y_pos, f1_scores, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fields, fontsize=7)
    ax.set_xlabel("F1 Score")
    ax.set_title(f"Lowest F1 Fields — {metrics.model_name}")
    ax.set_xlim(0, 1)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved per-field F1 chart: {output_path}")
