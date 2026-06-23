"""C3: Build Silver Label Dataset — high-confidence agreed predictions as pseudo-labels.

Takes B4 (SetFit) and B5 (SciBERT) predictions on real TACC data, filters to
records where both models agree and confidence exceeds a threshold, and outputs
a training JSONL that can be combined with synthetic data for retraining.

Usage:
    python scripts/build_silver_labels.py \
        --threshold 0.7 \
        --output data/generated/silver_labels.jsonl

    # Then retrain with combined data:
    python scripts/train_classifier.py --config configs/train.yaml \
        --train-path data/generated/train_with_silver.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_predictions(path: Path) -> list[dict]:
    """Load predictions from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data["predictions"]


def find_prediction_files(predictions_dir: Path, archive_dir: Path | None = None) -> dict[str, Path]:
    """Find B4 and B5 prediction files for real TACC."""
    files = {}
    for search_dir in [predictions_dir, archive_dir]:
        if search_dir is None or not search_dir.exists():
            continue
        for p in sorted(search_dir.glob("predictions_*real_tacc.json")):
            name = p.stem
            if "setfit" in name and "setfit" not in files:
                files["setfit"] = p
            elif "finetune" in name and "finetune" not in files:
                files["finetune"] = p
    return files


def build_silver_labels(
    preds_b4: list[dict],
    preds_b5: list[dict],
    threshold: float = 0.7,
    require_broad_agree: bool = True,
    exclude_unassigned: bool = True,
) -> tuple[list[dict], dict]:
    """Build silver label dataset from agreed high-confidence predictions.

    Args:
        preds_b4: SetFit predictions on real TACC.
        preds_b5: SciBERT predictions on real TACC.
        threshold: Minimum confidence for inclusion (max of B4, B5 confidence).
        require_broad_agree: Also require broad field agreement (recommended).
        exclude_unassigned: Exclude records with DB label "UNASSIGNED" or empty.

    Returns:
        (silver_records, stats_dict)
    """
    assert len(preds_b4) == len(preds_b5), "Prediction sets must have same size"

    silver_records = []
    excluded_records = []
    stats = {
        "total_input": len(preds_b4),
        "excluded_unassigned": 0,
        "models_disagree_major": 0,
        "models_disagree_broad": 0,
        "below_threshold": 0,
        "silver_labels": 0,
        "silver_matches_db": 0,
        "silver_disagrees_db": 0,
        "field_distribution": Counter(),
    }

    for p4, p5 in zip(preds_b4, preds_b5):
        db_label = p4.get("true_major_field", "")

        # Skip unassigned/empty
        if exclude_unassigned and (not db_label or db_label == "UNASSIGNED"):
            stats["excluded_unassigned"] += 1
            continue

        pred4 = p4["predicted_major_field"]
        pred5 = p5["predicted_major_field"]
        conf4 = p4.get("confidence", 0) or 0
        conf5 = p5.get("confidence", 0) or 0
        max_conf = max(conf4, conf5)

        # Models must agree on major field
        if pred4 != pred5:
            stats["models_disagree_major"] += 1
            continue

        # Optionally require broad field agreement too
        if require_broad_agree:
            broad4 = p4.get("predicted_broad_field", "")
            broad5 = p5.get("predicted_broad_field", "")
            if broad4 != broad5:
                stats["models_disagree_broad"] += 1
                continue

        # Confidence threshold
        if max_conf < threshold:
            stats["below_threshold"] += 1
            continue

        # This record qualifies as a silver label
        silver_record = {
            "abstract": p4["abstract"],
            "major_field": pred4,  # Use MODEL prediction, not DB label
            "broad_field": p4.get("predicted_broad_field", ""),
            "source": "silver_label",
            "confidence": max_conf,
            "conf_b4": conf4,
            "conf_b5": conf5,
            "db_major_field": db_label,  # Keep DB label for analysis
            "db_broad_field": p4.get("true_broad_field", ""),
        }
        silver_records.append(silver_record)
        stats["silver_labels"] += 1
        stats["field_distribution"][pred4] += 1

        if pred4 == db_label:
            stats["silver_matches_db"] += 1
        else:
            stats["silver_disagrees_db"] += 1

    # Convert Counter to dict for JSON serialization
    stats["field_distribution"] = dict(stats["field_distribution"])
    return silver_records, stats


def create_combined_training_set(
    synthetic_path: Path,
    silver_records: list[dict],
    output_path: Path,
    silver_weight: float = 1.0,
):
    """Combine synthetic training data with silver labels.

    Args:
        synthetic_path: Path to original synthetic train.jsonl.
        silver_records: Silver label records.
        output_path: Output path for combined JSONL.
        silver_weight: How many copies of silver labels to include (for upweighting).
    """
    # Load synthetic
    synthetic_records = []
    with open(synthetic_path) as f:
        for line in f:
            record = json.loads(line)
            record["source"] = "synthetic"
            synthetic_records.append(record)

    # Combine
    combined = list(synthetic_records)
    n_copies = max(1, int(silver_weight))
    for _ in range(n_copies):
        for r in silver_records:
            # Only include fields needed for training
            combined.append({
                "abstract": r["abstract"],
                "major_field": r["major_field"],
                "broad_field": r["broad_field"],
                "source": r["source"],
            })

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(combined)

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in combined:
            f.write(json.dumps(record) + "\n")

    return {
        "n_synthetic": len(synthetic_records),
        "n_silver": len(silver_records),
        "n_silver_copies": n_copies,
        "n_combined": len(combined),
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Build Silver Label Dataset")
    parser.add_argument("--predictions-dir", type=Path, default=Path("output/predictions"))
    parser.add_argument("--archive-dir", type=Path, default=Path("archive/output/predictions"))
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Confidence threshold for silver label inclusion")
    parser.add_argument("--output", type=Path, default=Path("data/generated/silver_labels.jsonl"))
    parser.add_argument("--synthetic-train", type=Path, default=Path("data/generated/train.jsonl"))
    parser.add_argument("--combined-output", type=Path, default=Path("data/generated/train_with_silver.jsonl"))
    parser.add_argument("--silver-weight", type=float, default=1.0,
                        help="Weight for silver labels (1.0 = equal to synthetic, 2.0 = double)")
    parser.add_argument("--no-combine", action="store_true",
                        help="Only output silver labels, don't create combined training set")
    args = parser.parse_args()

    # Find prediction files
    pred_files = find_prediction_files(args.predictions_dir, args.archive_dir)

    if "setfit" not in pred_files:
        print("ERROR: No SetFit prediction file found.")
        return
    if "finetune" not in pred_files:
        print("ERROR: No SciBERT finetune prediction file found.")
        return

    print(f"B4 (SetFit): {pred_files['setfit']}")
    print(f"B5 (SciBERT): {pred_files['finetune']}")
    print(f"Confidence threshold: {args.threshold}")

    # Load predictions
    preds_b4 = load_predictions(pred_files["setfit"])
    preds_b5 = load_predictions(pred_files["finetune"])

    # Build silver labels
    print("\nBuilding silver label dataset...")
    silver_records, stats = build_silver_labels(preds_b4, preds_b5, threshold=args.threshold)

    # Print stats
    print(f"\n--- Silver Label Stats ---")
    print(f"Total input records: {stats['total_input']}")
    print(f"Excluded (unassigned/empty): {stats['excluded_unassigned']}")
    print(f"Models disagree (major): {stats['models_disagree_major']}")
    print(f"Models disagree (broad): {stats['models_disagree_broad']}")
    print(f"Below confidence threshold: {stats['below_threshold']}")
    print(f"Silver labels created: {stats['silver_labels']}")
    print(f"  - Matches DB label: {stats['silver_matches_db']} "
          f"({stats['silver_matches_db'] / max(stats['silver_labels'], 1):.1%})")
    print(f"  - Disagrees with DB: {stats['silver_disagrees_db']} "
          f"({stats['silver_disagrees_db'] / max(stats['silver_labels'], 1):.1%})")

    # Field distribution
    print(f"\nTop 10 fields in silver set:")
    sorted_fields = sorted(stats["field_distribution"].items(), key=lambda x: x[1], reverse=True)
    for field, count in sorted_fields[:10]:
        print(f"  {field}: {count}")

    # Save silver labels
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for record in silver_records:
            f.write(json.dumps(record) + "\n")
    print(f"\nSaved silver labels to: {args.output}")

    # Save stats
    stats_path = args.output.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to: {stats_path}")

    # Create combined training set
    if not args.no_combine:
        if not args.synthetic_train.exists():
            print(f"\nWARNING: Synthetic training file not found at {args.synthetic_train}")
            print("  Skipping combined dataset creation. Use --synthetic-train to specify path.")
        else:
            print(f"\nCreating combined training set...")
            combine_stats = create_combined_training_set(
                args.synthetic_train, silver_records, args.combined_output,
                silver_weight=args.silver_weight,
            )
            print(f"  Synthetic records: {combine_stats['n_synthetic']}")
            print(f"  Silver records: {combine_stats['n_silver']} (×{combine_stats['n_silver_copies']})")
            print(f"  Combined total: {combine_stats['n_combined']}")
            print(f"  Saved to: {combine_stats['output_path']}")

    # Summary of what to do next
    print("\n--- Next Steps ---")
    print("1. Run audit_labels.py to visualize label quality")
    print("2. Review the silver labels — are they reasonable?")
    print("3. Retrain SetFit and SciBERT on combined data:")
    print(f"   python scripts/train_classifier.py --config configs/train.yaml \\")
    print(f"       --train-path {args.combined_output}")


if __name__ == "__main__":
    main()
