"""Step 3: Evaluate classification results against existing labels."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .utils import load_json, save_json


def run(cfg: PipelineConfig, project_root: Path) -> None:
    """Compute accuracy, per-field metrics, confusion analysis, confidence breakdown."""
    results_path = cfg.resolve_path(cfg.paths.classification_results, project_root)
    results = load_json(results_path)

    fields_path = cfg.resolve_path(cfg.paths.major_fields_json, project_root)
    valid_major_fields = sorted(load_json(fields_path))

    print(f"Loaded {len(results)} classification results")

    # Filter to records with valid existing labels
    labeled = [r for r in results if r["existing_label"] in valid_major_fields]
    print(f"Records with valid existing major field labels: {len(labeled)}")

    if not labeled:
        print("No labeled records to evaluate. Exiting.")
        return

    # Overall accuracy
    correct = sum(1 for r in labeled if r["predicted_field"] == r["existing_label"])
    accuracy = correct / len(labeled)
    print(f"\nOverall accuracy: {correct}/{len(labeled)} = {accuracy:.4f}")

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
    print(f"Overall accuracy: {accuracy:.4f}")
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
        "accuracy": round(accuracy, 4),
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
