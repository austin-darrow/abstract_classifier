"""Evaluation metrics: accuracy, per-field F1, confusion analysis.

Works with the standardized PredictionSet format (see predictions.py).
Legacy `run()` function retained for backward compat with FAISS baseline.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import PipelineConfig
from ..utils import load_json, save_json
from .predictions import Prediction, PredictionSet


# ---------------------------------------------------------------------------
# Core metrics computation on PredictionSet
# ---------------------------------------------------------------------------


@dataclass
class ClassificationMetrics:
    """Computed metrics for a set of predictions."""

    model_name: str
    dataset: str
    total: int
    major_field_accuracy: float
    broad_field_accuracy: float
    macro_f1_major: float
    macro_f1_broad: float
    per_field_metrics: dict  # field_name -> {precision, recall, f1, support}
    top_k_accuracy: dict[int, float] = field(default_factory=dict)  # k -> accuracy
    confusion_pairs: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def summary_row(self) -> dict:
        """Single-row dict for comparison tables."""
        row = {
            "model": self.model_name,
            "dataset": self.dataset,
            "n": self.total,
            "major_acc": self.major_field_accuracy,
            "broad_acc": self.broad_field_accuracy,
            "macro_f1_major": self.macro_f1_major,
            "macro_f1_broad": self.macro_f1_broad,
        }
        for k, v in self.top_k_accuracy.items():
            row[f"top{k}_major_acc"] = v
        return row

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ClassificationMetrics":
        import json
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def compute_metrics(
    pred_set: PredictionSet,
    top_k_values: tuple[int, ...] = (3, 5),
    n_confused_pairs: int = 20,
) -> ClassificationMetrics:
    """Compute all metrics from a PredictionSet."""
    preds = pred_set.predictions
    n = len(preds)
    if n == 0:
        return ClassificationMetrics(
            model_name=pred_set.model_name, dataset=pred_set.dataset,
            total=0, major_field_accuracy=0, broad_field_accuracy=0,
            macro_f1_major=0, macro_f1_broad=0, per_field_metrics={},
        )

    # Major field accuracy
    major_correct = sum(
        1 for p in preds if p.predicted_major_field == p.true_major_field
    )
    major_acc = major_correct / n

    # Broad field accuracy
    broad_correct = sum(
        1 for p in preds if p.predicted_broad_field == p.true_broad_field
    )
    broad_acc = broad_correct / n

    # Top-k accuracy (major field)
    top_k_acc = {}
    for k in top_k_values:
        hits = sum(
            1 for p in preds
            if p.true_major_field in p.top_k_major_fields[:k]
        )
        top_k_acc[k] = hits / n

    # Per-field precision/recall/F1 (major)
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    for p in preds:
        if p.predicted_major_field == p.true_major_field:
            tp[p.true_major_field] += 1
        else:
            fp[p.predicted_major_field] += 1
            fn[p.true_major_field] += 1

    all_fields = set(tp.keys()) | set(fp.keys()) | set(fn.keys())
    per_field = {}
    f1_values = []
    for fld in sorted(all_fields):
        prec = tp[fld] / (tp[fld] + fp[fld]) if (tp[fld] + fp[fld]) > 0 else 0
        rec = tp[fld] / (tp[fld] + fn[fld]) if (tp[fld] + fn[fld]) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        support = tp[fld] + fn[fld]
        if support > 0:
            per_field[fld] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": support,
            }
            f1_values.append(f1)

    macro_f1_major = float(np.mean(f1_values)) if f1_values else 0.0

    # Per-field F1 for broad fields
    tp_b: Counter = Counter()
    fp_b: Counter = Counter()
    fn_b: Counter = Counter()
    for p in preds:
        if p.predicted_broad_field == p.true_broad_field:
            tp_b[p.true_broad_field] += 1
        else:
            fp_b[p.predicted_broad_field] += 1
            fn_b[p.true_broad_field] += 1
    all_broad = set(tp_b.keys()) | set(fp_b.keys()) | set(fn_b.keys())
    f1_broad_values = []
    for fld in all_broad:
        prec = tp_b[fld] / (tp_b[fld] + fp_b[fld]) if (tp_b[fld] + fp_b[fld]) > 0 else 0
        rec = tp_b[fld] / (tp_b[fld] + fn_b[fld]) if (tp_b[fld] + fn_b[fld]) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if (tp_b[fld] + fn_b[fld]) > 0:
            f1_broad_values.append(f1)
    macro_f1_broad = float(np.mean(f1_broad_values)) if f1_broad_values else 0.0

    # Top confused pairs (major field)
    confusion: Counter = Counter()
    for p in preds:
        if p.predicted_major_field != p.true_major_field:
            confusion[(p.true_major_field, p.predicted_major_field)] += 1
    top_confused = [
        {"true_field": pair[0], "predicted_field": pair[1], "count": count}
        for pair, count in confusion.most_common(n_confused_pairs)
    ]

    return ClassificationMetrics(
        model_name=pred_set.model_name,
        dataset=pred_set.dataset,
        total=n,
        major_field_accuracy=round(major_acc, 4),
        broad_field_accuracy=round(broad_acc, 4),
        macro_f1_major=round(macro_f1_major, 4),
        macro_f1_broad=round(macro_f1_broad, 4),
        per_field_metrics=per_field,
        top_k_accuracy={k: round(v, 4) for k, v in top_k_acc.items()},
        confusion_pairs=top_confused,
    )


def print_metrics(metrics: ClassificationMetrics) -> None:
    """Print a summary of classification metrics."""
    print(f"\n{'='*60}")
    print(f"EVALUATION: {metrics.model_name} on {metrics.dataset}")
    print(f"{'='*60}")
    print(f"Total predictions: {metrics.total}")
    print(f"Major field accuracy: {metrics.major_field_accuracy:.4f}")
    print(f"Broad field accuracy: {metrics.broad_field_accuracy:.4f}")
    print(f"Macro F1 (major): {metrics.macro_f1_major:.4f}")
    print(f"Macro F1 (broad): {metrics.macro_f1_broad:.4f}")
    for k, v in sorted(metrics.top_k_accuracy.items()):
        print(f"Top-{k} major accuracy: {v:.4f}")
    if metrics.confusion_pairs:
        print(f"\nTop confused pairs:")
        for item in metrics.confusion_pairs[:10]:
            print(f"  {item['true_field'][:40]:40s} -> {item['predicted_field'][:40]:40s} ({item['count']})")
    if metrics.per_field_metrics:
        sorted_fields = sorted(
            metrics.per_field_metrics.items(), key=lambda x: x[1]["f1"]
        )
        print(f"\nLowest F1 fields (bottom 10):")
        for fld, m in sorted_fields[:10]:
            print(f"  {fld[:50]:50s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} (n={m['support']})")


# ---------------------------------------------------------------------------
# Legacy: run() for backward compat with FAISS baseline output format
# ---------------------------------------------------------------------------


def _build_major_to_broad_mapping(taxonomy: list[dict]) -> dict[str, str]:
    """Build a mapping from major field label to broad field label."""
    mapping = {}
    for entry in taxonomy:
        major = entry["Major_Field_label"]
        broad = entry["Broad_Field_label"]
        mapping.setdefault(major, broad)
    return mapping


def run(cfg: PipelineConfig, project_root: Path) -> None:
    """Compute accuracy, per-field metrics, confusion analysis, confidence breakdown."""
    results_path = cfg.resolve_path(cfg.paths.classification_results, project_root)
    results = load_json(results_path)

    fields_path = cfg.resolve_path(cfg.paths.major_fields_json, project_root)
    valid_major_fields = sorted(load_json(fields_path))

    broad_fields_path = cfg.resolve_path(cfg.paths.broad_fields_json, project_root)
    valid_broad_fields = sorted(load_json(broad_fields_path))

    # Build major→broad mapping from taxonomy
    taxonomy_path = cfg.resolve_path(cfg.paths.taxonomy_json, project_root)
    taxonomy = load_json(taxonomy_path)
    major_to_broad = _build_major_to_broad_mapping(taxonomy)

    print(f"Loaded {len(results)} classification results")

    # Filter to records with valid existing labels
    labeled = [r for r in results if r["existing_label"] in valid_major_fields]
    print(f"Records with valid existing major field labels: {len(labeled)}")

    if not labeled:
        print("No labeled records to evaluate. Exiting.")
        return

    # Overall accuracy (major field)
    correct = sum(1 for r in labeled if r["predicted_field"] == r["existing_label"])
    accuracy = correct / len(labeled)
    print(f"\nMajor field accuracy: {correct}/{len(labeled)} = {accuracy:.4f}")

    # Broad field accuracy
    broad_correct = 0
    broad_labeled_count = 0
    for r in labeled:
        existing_broad = major_to_broad.get(r["existing_label"])
        predicted_broad = r.get("predicted_broad_field") or major_to_broad.get(r["predicted_field"])
        if existing_broad:
            broad_labeled_count += 1
            if predicted_broad == existing_broad:
                broad_correct += 1

    broad_accuracy = broad_correct / broad_labeled_count if broad_labeled_count > 0 else 0
    print(f"Broad field accuracy: {broad_correct}/{broad_labeled_count} = {broad_accuracy:.4f}")

    # Per-field metrics (precision, recall, F1)
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()

    for r in labeled:
        pred = r["predicted_field"]
        true = r["existing_label"]
        if pred == true:
            tp[pred] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    per_field_metrics = {}
    for field in valid_major_fields:
        p = tp[field] / (tp[field] + fp[field]) if (tp[field] + fp[field]) > 0 else 0
        r = tp[field] / (tp[field] + fn[field]) if (tp[field] + fn[field]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        support = tp[field] + fn[field]
        if support > 0:
            per_field_metrics[field] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "support": support,
            }

    # Confusion analysis: top confused pairs
    confusion_pairs: Counter = Counter()
    for r in labeled:
        if r["predicted_field"] != r["existing_label"]:
            pair = (r["existing_label"], r["predicted_field"])
            confusion_pairs[pair] += 1

    n_confused = cfg.evaluate.top_confused_pairs
    top_confused = [
        {"true_field": pair[0], "predicted_field": pair[1], "count": count}
        for pair, count in confusion_pairs.most_common(n_confused)
    ]

    # Confidence breakdown
    thresholds = cfg.evaluate.confidence_thresholds
    confidence_breakdown = {}

    for thresh in thresholds:
        subset = [r for r in labeled if r["agreement_ratio"] >= thresh]
        if subset:
            acc = sum(1 for r in subset if r["predicted_field"] == r["existing_label"]) / len(subset)
            confidence_breakdown[f"agreement>={thresh}"] = {
                "count": len(subset),
                "accuracy": round(acc, 4),
            }

    low_conf = [r for r in labeled if r["agreement_ratio"] < 0.5]
    if low_conf:
        acc = sum(1 for r in low_conf if r["predicted_field"] == r["existing_label"]) / len(low_conf)
        confidence_breakdown["agreement<0.5"] = {
            "count": len(low_conf),
            "accuracy": round(acc, 4),
        }

    # Similarity stats
    sim_stats = {
        "mean_top1_similarity": round(np.mean([r["top1_similarity"] for r in labeled]), 4),
        "median_top1_similarity": round(float(np.median([r["top1_similarity"] for r in labeled])), 4),
        "correct_mean_sim": round(
            np.mean([r["top1_similarity"] for r in labeled if r["predicted_field"] == r["existing_label"]]) if correct > 0 else 0, 4
        ),
        "incorrect_mean_sim": round(
            np.mean([r["top1_similarity"] for r in labeled if r["predicted_field"] != r["existing_label"]]) if (len(labeled) - correct) > 0 else 0, 4
        ),
    }

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total labeled records: {len(labeled)}")
    print(f"Major field accuracy: {accuracy:.4f}")
    print(f"Broad field accuracy: {broad_accuracy:.4f}")
    print(f"\nConfidence breakdown:")
    for key, val in confidence_breakdown.items():
        print(f"  {key}: {val['count']} records, accuracy={val['accuracy']:.4f}")
    print(f"\nSimilarity stats:")
    for key, val in sim_stats.items():
        print(f"  {key}: {val}")
    print(f"\nTop 10 confused field pairs:")
    for item in top_confused[:10]:
        print(f"  {item['true_field'][:40]:40s} -> {item['predicted_field'][:40]:40s} ({item['count']})")

    sorted_fields = sorted(per_field_metrics.items(), key=lambda x: x[1]["f1"])
    print(f"\nLowest F1 fields (bottom 10):")
    for field, m in sorted_fields[:10]:
        print(f"  {field[:50]:50s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} (n={m['support']})")

    # Save outputs
    evaluation_report = {
        "total_records": len(results),
        "labeled_records": len(labeled),
        "major_field_accuracy": round(accuracy, 4),
        "broad_field_accuracy": round(broad_accuracy, 4),
        "confidence_breakdown": confidence_breakdown,
        "similarity_stats": sim_stats,
        "per_field_metrics": per_field_metrics,
        "top_confused_pairs": top_confused,
    }

    report_path = cfg.resolve_path(cfg.paths.evaluation_report, project_root)
    save_json(evaluation_report, report_path)

    # Save disagreements for manual review
    disagreements = [
        {
            "abstract": r["abstract"][:500],
            "existing_label": r["existing_label"],
            "predicted_field": r["predicted_field"],
            "top1_similarity": r["top1_similarity"],
            "agreement_ratio": r["agreement_ratio"],
            "top1_detailed_field": r["top1_detailed_field"],
            "top1_cip_title": r["top1_cip_title"],
            "top10_fields": r["top10_fields"],
        }
        for r in labeled
        if r["predicted_field"] != r["existing_label"]
    ]

    disagreements_path = cfg.resolve_path(cfg.paths.disagreements, project_root)
    save_json(disagreements, disagreements_path)

    print(f"\nSaved: {report_path}, {disagreements_path} ({len(disagreements)} disagreements)")
    print("Done.")
