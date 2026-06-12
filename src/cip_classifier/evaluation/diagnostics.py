"""Training diagnostics: learning curves, per-field loss analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .metrics import compute_metrics
from .predictions import PredictionSet


def learning_curve(
    train_fn: Callable[[list, float], PredictionSet],
    train_data: list,
    fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0),
    seed: int = 42,
) -> list[dict]:
    """Run a learning curve experiment.

    Args:
        train_fn: Function that takes (train_subset, fraction) and returns
                  a PredictionSet evaluated on a held-out test set.
        train_data: Full training data (list of dicts or whatever train_fn expects).
        fractions: Fractions of training data to use.
        seed: Random seed for subset selection.

    Returns:
        List of dicts with {fraction, n_train, metrics_dict}.
    """
    rng = np.random.default_rng(seed)
    n = len(train_data)
    results = []

    for frac in fractions:
        k = max(1, int(n * frac))
        indices = rng.choice(n, size=k, replace=False)
        subset = [train_data[i] for i in indices]

        pred_set = train_fn(subset, frac)
        metrics = compute_metrics(pred_set)

        results.append({
            "fraction": frac,
            "n_train": k,
            "major_field_accuracy": metrics.major_field_accuracy,
            "broad_field_accuracy": metrics.broad_field_accuracy,
            "macro_f1_major": metrics.macro_f1_major,
        })
        print(f"  fraction={frac:.2f} (n={k}): major_acc={metrics.major_field_accuracy:.4f}, broad_acc={metrics.broad_field_accuracy:.4f}")

    return results


def plot_learning_curve(
    results: list[dict],
    output_path: Path,
    title: str = "Learning Curve",
) -> None:
    """Plot learning curve from results of learning_curve()."""
    import matplotlib.pyplot as plt

    fractions = [r["fraction"] for r in results]
    major_acc = [r["major_field_accuracy"] for r in results]
    broad_acc = [r["broad_field_accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fractions, major_acc, "o-", label="Major Field Acc")
    ax.plot(fractions, broad_acc, "s-", label="Broad Field Acc")
    ax.axhline(y=0.7, color="r", linestyle="--", alpha=0.5, label="Major target")
    ax.axhline(y=0.9, color="g", linestyle="--", alpha=0.5, label="Broad target")

    ax.set_xlabel("Fraction of Training Data")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved learning curve: {output_path}")


def per_field_error_analysis(pred_set: PredictionSet) -> dict:
    """Analyze which fields are hardest, where errors cluster.

    Returns:
        Dict with per-field accuracy and common error patterns.
    """
    from collections import Counter, defaultdict

    field_correct: Counter = Counter()
    field_total: Counter = Counter()
    field_errors: dict[str, Counter] = defaultdict(Counter)

    for p in pred_set.predictions:
        field_total[p.true_major_field] += 1
        if p.predicted_major_field == p.true_major_field:
            field_correct[p.true_major_field] += 1
        else:
            field_errors[p.true_major_field][p.predicted_major_field] += 1

    analysis = {}
    for field in sorted(field_total.keys()):
        total = field_total[field]
        correct = field_correct[field]
        acc = correct / total if total > 0 else 0
        top_errors = field_errors[field].most_common(3)
        analysis[field] = {
            "accuracy": round(acc, 4),
            "total": total,
            "correct": correct,
            "top_confusions": [
                {"predicted": pred, "count": cnt} for pred, cnt in top_errors
            ],
        }

    return analysis


def cross_broad_field_errors(pred_set: PredictionSet) -> dict:
    """Analyze errors that cross broad-field boundaries vs within same broad field."""
    within = 0
    across = 0
    across_pairs: dict = {}

    for p in pred_set.predictions:
        if p.predicted_major_field != p.true_major_field:
            if p.predicted_broad_field == p.true_broad_field:
                within += 1
            else:
                across += 1
                key = f"{p.true_broad_field} -> {p.predicted_broad_field}"
                across_pairs[key] = across_pairs.get(key, 0) + 1

    total_errors = within + across
    return {
        "total_errors": total_errors,
        "within_broad_field": within,
        "across_broad_field": across,
        "pct_across": round(across / total_errors, 4) if total_errors > 0 else 0,
        "top_cross_broad_pairs": sorted(
            across_pairs.items(), key=lambda x: -x[1]
        )[:10],
    }
