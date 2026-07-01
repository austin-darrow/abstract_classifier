"""Compare multiple classifier approaches side-by-side."""

from __future__ import annotations

from pathlib import Path

from .metrics import ClassificationMetrics, compute_metrics, print_metrics
from .predictions import PredictionSet


def load_all_predictions(results_dir: Path) -> list[PredictionSet]:
    """Load all prediction JSON files from a directory.

    Skips files that don't follow the standard PredictionSet format
    (e.g. *_clean_eval.json which are evaluation reports, not prediction sets).
    """
    pred_sets = []
    for path in sorted(results_dir.glob("predictions_*.json")):
        if "_clean_eval" in path.name:
            continue
        pred_sets.append(PredictionSet.load(path))
    return pred_sets


def compare(pred_sets: list[PredictionSet], top_k_values: tuple[int, ...] = (3, 5)) -> list[ClassificationMetrics]:
    """Compute metrics for each prediction set and return them."""
    return [compute_metrics(ps, top_k_values=top_k_values) for ps in pred_sets]


def comparison_table(metrics_list: list[ClassificationMetrics]) -> str:
    """Format a comparison table as a string."""
    if not metrics_list:
        return "No results to compare."

    rows = [m.summary_row() for m in metrics_list]
    # Determine columns
    cols = list(rows[0].keys())

    # Column widths
    widths = {c: max(len(c), max(len(_fmt(r.get(c, ""))) for r in rows)) for c in cols}

    # Header
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    lines = [header, sep]

    # Rows
    for row in rows:
        line = " | ".join(_fmt(row.get(c, "")).ljust(widths[c]) for c in cols)
        lines.append(line)

    return "\n".join(lines)


def agreement_analysis(pred_sets: list[PredictionSet]) -> dict:
    """Analyze where models agree/disagree.

    Returns dict with:
      - unanimous: count where all models agree (and correctness)
      - majority: count where majority agrees
      - disagreement_examples: sample abstracts where models diverge
    """
    if len(pred_sets) < 2:
        return {"error": "Need at least 2 prediction sets"}

    # Align predictions by abstract text
    n = min(ps.size for ps in pred_sets)
    unanimous_correct = 0
    unanimous_wrong = 0
    majority_correct = 0
    disagreements = []

    for i in range(n):
        preds_i = [ps.predictions[i].predicted_major_field for ps in pred_sets]
        true = pred_sets[0].predictions[i].true_major_field
        unique = set(preds_i)

        if len(unique) == 1:
            if preds_i[0] == true:
                unanimous_correct += 1
            else:
                unanimous_wrong += 1
        else:
            # Check majority
            from collections import Counter
            counts = Counter(preds_i)
            majority_pred, majority_count = counts.most_common(1)[0]
            if majority_pred == true:
                majority_correct += 1
            if len(disagreements) < 20:
                disagreements.append({
                    "abstract": pred_sets[0].predictions[i].abstract[:200],
                    "true": true,
                    "predictions": {
                        ps.model_name: ps.predictions[i].predicted_major_field
                        for ps in pred_sets
                    },
                })

    return {
        "n_compared": n,
        "unanimous_correct": unanimous_correct,
        "unanimous_wrong": unanimous_wrong,
        "majority_correct": majority_correct,
        "n_disagreements": n - unanimous_correct - unanimous_wrong,
        "disagreement_examples": disagreements,
    }


def run(results_dir: Path, output_dir: Path, top_k_values: tuple[int, ...] = (3, 5)) -> None:
    """Load all predictions, compute metrics, print comparison, save results."""
    pred_sets = load_all_predictions(results_dir)
    if not pred_sets:
        print(f"No prediction files found in {results_dir}")
        return

    print(f"Found {len(pred_sets)} prediction sets")
    metrics_list = compare(pred_sets, top_k_values=top_k_values)

    # Print each model's summary
    for m in metrics_list:
        print_metrics(m)

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(comparison_table(metrics_list))

    # Save individual metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    for m in metrics_list:
        safe_name = m.model_name.replace("/", "_").replace(" ", "_")
        m.save(output_dir / f"metrics_{safe_name}_{m.dataset}.json")

    # Save comparison summary
    import json
    summary = {
        "models": [m.summary_row() for m in metrics_list],
    }
    if len(pred_sets) >= 2:
        summary["agreement"] = agreement_analysis(pred_sets)

    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved comparison results to {output_dir}")


def _fmt(val) -> str:
    """Format a value for table display."""
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)
