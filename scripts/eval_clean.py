"""Evaluate model predictions against a "clean" subset of real TACC labels.

Since DB labels have ~55% noise rate (per audit), standard evaluation is unreliable.
This script provides multiple evaluation strategies:

1. "clean" — Only evaluate on records where BOTH B4 and B5 agree with the DB label
   (i.e., records where we're confident the DB label is correct).
2. "consensus" — Evaluate how often the new model agrees with B4/B5 consensus
   (measures self-consistency, not accuracy against DB).
3. "filtered" — Exclude UNASSIGNED and known-bad fields from evaluation.

Usage:
    python scripts/eval_clean.py \
        --target output/predictions/predictions_finetune_scibert_scivocab_uncased_5ep_real_tacc.json \
        --reference-b4 output/predictions/predictions_setfit_20iter_1ep_real_tacc.json \
        --reference-b5 output/predictions/predictions_finetune_scibert_scivocab_uncased_3ep_real_tacc.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load_predictions(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["predictions"]


def eval_clean_subset(target_preds: list[dict], ref_b4: list[dict], ref_b5: list[dict]) -> dict:
    """Evaluate only on records where B4, B5, and DB label all agree (trusted labels)."""
    assert len(target_preds) == len(ref_b4) == len(ref_b5)

    clean_indices = []
    for i in range(len(ref_b4)):
        db_label = ref_b4[i]["true_major_field"]
        if not db_label or db_label == "UNASSIGNED":
            continue
        pred_b4 = ref_b4[i]["predicted_major_field"]
        pred_b5 = ref_b5[i]["predicted_major_field"]
        # All three agree — high confidence this label is correct
        if pred_b4 == db_label and pred_b5 == db_label:
            clean_indices.append(i)

    if not clean_indices:
        return {"n": 0, "error": "No clean records found"}

    correct_major = 0
    correct_broad = 0
    field_correct = Counter()
    field_total = Counter()

    for i in clean_indices:
        true_major = ref_b4[i]["true_major_field"]
        true_broad = ref_b4[i]["true_broad_field"]
        pred_major = target_preds[i]["predicted_major_field"]
        pred_broad = target_preds[i]["predicted_broad_field"]

        field_total[true_major] += 1
        if pred_major == true_major:
            correct_major += 1
            field_correct[true_major] += 1
        if pred_broad == true_broad:
            correct_broad += 1

    n = len(clean_indices)
    per_field = {}
    for field in sorted(field_total.keys()):
        acc = field_correct[field] / field_total[field]
        per_field[field] = {"accuracy": round(acc, 4), "n": field_total[field]}

    return {
        "n": n,
        "major_accuracy": round(correct_major / n, 4),
        "broad_accuracy": round(correct_broad / n, 4),
        "per_field": per_field,
    }


def eval_consensus(target_preds: list[dict], ref_b4: list[dict], ref_b5: list[dict],
                   confidence_threshold: float = 0.7) -> dict:
    """Measure how often the target model agrees with B4/B5 consensus.

    This doesn't measure 'accuracy' — it measures whether the retrained model
    produces the same predictions as the reference models when they agree.
    Useful for measuring whether silver-label retraining reinforced consensus.
    """
    assert len(target_preds) == len(ref_b4) == len(ref_b5)

    consensus_indices = []
    for i in range(len(ref_b4)):
        db_label = ref_b4[i]["true_major_field"]
        if not db_label or db_label == "UNASSIGNED":
            continue
        pred_b4 = ref_b4[i]["predicted_major_field"]
        pred_b5 = ref_b5[i]["predicted_major_field"]
        conf_b4 = ref_b4[i].get("confidence", 0) or 0
        conf_b5 = ref_b5[i].get("confidence", 0) or 0
        max_conf = max(conf_b4, conf_b5)

        if pred_b4 == pred_b5 and max_conf >= confidence_threshold:
            consensus_indices.append(i)

    if not consensus_indices:
        return {"n": 0, "error": "No consensus records found"}

    agrees_with_consensus = 0
    agrees_with_db = 0
    consensus_agrees_with_db = 0

    for i in consensus_indices:
        consensus_label = ref_b4[i]["predicted_major_field"]
        db_label = ref_b4[i]["true_major_field"]
        target_pred = target_preds[i]["predicted_major_field"]

        if target_pred == consensus_label:
            agrees_with_consensus += 1
        if target_pred == db_label:
            agrees_with_db += 1
        if consensus_label == db_label:
            consensus_agrees_with_db += 1

    n = len(consensus_indices)
    return {
        "n": n,
        "target_agrees_with_consensus": round(agrees_with_consensus / n, 4),
        "target_agrees_with_db": round(agrees_with_db / n, 4),
        "consensus_agrees_with_db": round(consensus_agrees_with_db / n, 4),
    }


def eval_filtered(target_preds: list[dict], exclude_fields: set[str] | None = None) -> dict:
    """Standard eval but excluding UNASSIGNED and optionally known-bad fields."""
    if exclude_fields is None:
        exclude_fields = {"UNASSIGNED"}

    filtered = [p for p in target_preds
                if p["true_major_field"]
                and p["true_major_field"] not in exclude_fields]

    if not filtered:
        return {"n": 0, "error": "No records after filtering"}

    n = len(filtered)
    correct_major = sum(1 for p in filtered if p["predicted_major_field"] == p["true_major_field"])
    correct_broad = sum(1 for p in filtered if p["predicted_broad_field"] == p["true_broad_field"])

    return {
        "n": n,
        "major_accuracy": round(correct_major / n, 4),
        "broad_accuracy": round(correct_broad / n, 4),
        "excluded_fields": sorted(exclude_fields),
    }


def main():
    parser = argparse.ArgumentParser(description="Clean Evaluation (noise-aware)")
    parser.add_argument("--target", type=Path, required=True,
                        help="Predictions file to evaluate (the model you want to measure)")
    parser.add_argument("--reference-b4", type=Path, default=None,
                        help="B4 (SetFit) reference predictions for consensus checks")
    parser.add_argument("--reference-b5", type=Path, default=None,
                        help="B5 (SciBERT) reference predictions for consensus checks")
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                        help="Confidence threshold for consensus evaluation")
    parser.add_argument("--exclude-fields", type=str, nargs="*",
                        default=["UNASSIGNED"],
                        help="Fields to exclude from filtered evaluation")
    args = parser.parse_args()

    target_preds = load_predictions(args.target)
    print(f"Target: {args.target.name} ({len(target_preds)} predictions)")

    # Strategy 1: Filtered eval (always available)
    print(f"\n{'='*60}")
    print("STRATEGY 1: Filtered (exclude UNASSIGNED)")
    print(f"{'='*60}")
    filtered_result = eval_filtered(target_preds, set(args.exclude_fields))
    print(f"  Records evaluated: {filtered_result['n']}")
    print(f"  Major accuracy: {filtered_result.get('major_accuracy', 'N/A')}")
    print(f"  Broad accuracy: {filtered_result.get('broad_accuracy', 'N/A')}")

    # Strategy 2 & 3: Need reference predictions
    if args.reference_b4 and args.reference_b5:
        ref_b4 = load_predictions(args.reference_b4)
        ref_b5 = load_predictions(args.reference_b5)

        print(f"\n{'='*60}")
        print("STRATEGY 2: Clean subset (B4 + B5 + DB all agree = trusted labels)")
        print(f"{'='*60}")
        clean_result = eval_clean_subset(target_preds, ref_b4, ref_b5)
        print(f"  Records evaluated: {clean_result['n']}")
        print(f"  Major accuracy: {clean_result.get('major_accuracy', 'N/A')}")
        print(f"  Broad accuracy: {clean_result.get('broad_accuracy', 'N/A')}")

        if clean_result.get("per_field"):
            # Show worst fields
            per_field = clean_result["per_field"]
            sorted_fields = sorted(per_field.items(), key=lambda x: x[1]["accuracy"])
            print(f"\n  Lowest accuracy fields (clean subset):")
            for field, stats in sorted_fields[:10]:
                print(f"    {field:<50} acc={stats['accuracy']:.2f} (n={stats['n']})")

        print(f"\n{'='*60}")
        print(f"STRATEGY 3: Consensus agreement (B4+B5 agree at conf≥{args.confidence_threshold})")
        print(f"{'='*60}")
        consensus_result = eval_consensus(target_preds, ref_b4, ref_b5,
                                          confidence_threshold=args.confidence_threshold)
        print(f"  Consensus records: {consensus_result['n']}")
        print(f"  Target agrees with consensus: {consensus_result.get('target_agrees_with_consensus', 'N/A')}")
        print(f"  Target agrees with DB label: {consensus_result.get('target_agrees_with_db', 'N/A')}")
        print(f"  Consensus agrees with DB: {consensus_result.get('consensus_agrees_with_db', 'N/A')}")

    # Save results
    output_path = args.target.with_name(args.target.stem + "_clean_eval.json")
    results = {
        "target": str(args.target),
        "filtered": filtered_result,
    }
    if args.reference_b4 and args.reference_b5:
        results["clean_subset"] = clean_result
        results["consensus"] = consensus_result
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
