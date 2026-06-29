"""Hierarchical inference via constrained decoding (Option C).

Combines two pre-trained models at inference time:
  - Stage 1: Major-field model (74 classes, 93.96% accuracy)
  - Stage 2: Detailed-field model (315 classes, 87.94% accuracy)
    with logits masked to only detailed fields within the predicted major field.

Four scoring strategies are evaluated:
  A) Top-1 mask: Use top-1 major prediction, mask detailed logits to that major.
  B) Top-k max: For top-k major predictions, pick the (major, detailed) pair
     with highest detailed confidence after masking.
  C) Combined score: For top-k major predictions, score = major_prob * detailed_prob,
     pick the highest combined score.
  D) Weighted combined: score = major_prob^alpha * detailed_prob, with alpha
     controlling how much to trust the major model (default: 2.0, since the
     major model is more reliable).

Usage:
    python scripts/run_hierarchical_inference.py

    # Custom model paths:
    python scripts/run_hierarchical_inference.py \
        --major-model output/sweep/models/scibert_scivocab_uncased_lr3e-05_ep8_bs16_ls0.0_wd0.01_linear \
        --detailed-model output/models/detailed_finetune

    # Predict-only (skip comparisons, just output predictions):
    python scripts/run_hierarchical_inference.py --strategy combined
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def find_project_root() -> Path:
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return cwd


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_taxonomy_mappings(taxonomy_path: Path) -> tuple[dict, dict, dict]:
    """Build mappings from SEDCIP24.json.

    Returns:
        (detailed_to_major, detailed_to_broad, major_to_detailed_set)
    """
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)

    detailed_to_major = {}
    detailed_to_broad = {}
    major_to_detailed: dict[str, set] = defaultdict(set)

    for entry in taxonomy:
        d = entry["Detailed_Field_label"]
        m = entry["Major_Field_label"]
        b = entry["Broad_Field_label"]
        detailed_to_major.setdefault(d, m)
        detailed_to_broad.setdefault(d, b)
        major_to_detailed[m].add(d)

    return detailed_to_major, detailed_to_broad, dict(major_to_detailed)


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def load_model_and_predict(model_dir: Path, test_texts: list[str], batch_size: int = 32, max_length: int = 512):
    """Load a saved model and return raw logits for all test texts."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    with open(model_dir / "label_classes.json") as f:
        label_classes = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")

    ds = Dataset.from_dict({"text": test_texts}).map(tokenize_fn, batched=True)
    ds.set_format("torch", columns=["input_ids", "attention_mask"])

    all_logits = []
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            batch = ds[i:i + batch_size]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    return logits, label_classes


def build_major_to_detailed_indices(major_classes: list[str], detailed_classes: list[str],
                                     major_to_detailed_set: dict[str, set]) -> dict[int, list[int]]:
    """For each major class index, find which detailed class indices belong to it."""
    major_idx_to_detailed_idx: dict[int, list[int]] = {}
    for mi, major in enumerate(major_classes):
        detailed_set = major_to_detailed_set.get(major, set())
        indices = [di for di, d in enumerate(detailed_classes) if d in detailed_set]
        major_idx_to_detailed_idx[mi] = indices
    return major_idx_to_detailed_idx


def hierarchical_predict(
    major_logits: np.ndarray,
    detailed_logits: np.ndarray,
    major_classes: list[str],
    detailed_classes: list[str],
    major_to_detailed_indices: dict[int, list[int]],
    detailed_to_major: dict[str, str],
    detailed_to_broad: dict[str, str],
    top_k_major: int = 5,
    major_alpha: float = 2.0,
) -> dict[str, list[dict]]:
    """Run all four hierarchical strategies.

    Returns dict with keys 'top1_mask', 'topk_max', 'combined', 'weighted'
    each mapping to a list of prediction dicts.
    """
    n = len(major_logits)
    major_probs = softmax(major_logits)

    results = {"top1_mask": [], "topk_max": [], "combined": [], "weighted": []}

    for i in range(n):
        # Major model: get top-k
        major_sorted = np.argsort(major_probs[i])[::-1]
        top_k_majors = major_sorted[:top_k_major]

        # Strategy A: Top-1 mask
        best_major_idx = top_k_majors[0]
        pred_a = _mask_and_predict(
            detailed_logits[i], major_classes[best_major_idx],
            best_major_idx, major_to_detailed_indices, detailed_classes,
            detailed_to_major, detailed_to_broad,
        )
        pred_a["major_confidence"] = float(major_probs[i][best_major_idx])
        results["top1_mask"].append(pred_a)

        # Strategy B, C, D: consider top-k majors
        best_b = None
        best_b_score = -1.0
        best_c = None
        best_c_score = -1.0
        best_d = None
        best_d_score = -1.0

        for mi in top_k_majors:
            major_name = major_classes[mi]
            major_prob = float(major_probs[i][mi])

            pred = _mask_and_predict(
                detailed_logits[i], major_name, mi,
                major_to_detailed_indices, detailed_classes,
                detailed_to_major, detailed_to_broad,
            )
            pred["major_confidence"] = major_prob
            detailed_conf = pred["confidence"]

            # Strategy B: highest detailed confidence
            if detailed_conf > best_b_score:
                best_b_score = detailed_conf
                best_b = pred.copy()

            # Strategy C: combined score (equal weight)
            combined = major_prob * detailed_conf
            if combined > best_c_score:
                best_c_score = combined
                best_c = pred.copy()
                best_c["combined_score"] = combined

            # Strategy D: weighted combined (trust major model more)
            weighted = (major_prob ** major_alpha) * detailed_conf
            if weighted > best_d_score:
                best_d_score = weighted
                best_d = pred.copy()
                best_d["combined_score"] = weighted

        results["topk_max"].append(best_b)
        results["combined"].append(best_c)
        results["weighted"].append(best_d)

    return results


def _mask_and_predict(
    detailed_logits_row: np.ndarray,
    major_name: str,
    major_idx: int,
    major_to_detailed_indices: dict[int, list[int]],
    detailed_classes: list[str],
    detailed_to_major: dict[str, str],
    detailed_to_broad: dict[str, str],
) -> dict:
    """Mask detailed logits to a single major field and return best prediction."""
    valid_indices = major_to_detailed_indices.get(major_idx, [])

    if not valid_indices:
        # Fallback: no mapping found, use argmax of full logits
        best_idx = int(np.argmax(detailed_logits_row))
        probs_full = softmax(detailed_logits_row.reshape(1, -1))[0]
        return {
            "predicted_detailed_field": detailed_classes[best_idx],
            "predicted_major_field": detailed_to_major.get(detailed_classes[best_idx], ""),
            "predicted_broad_field": detailed_to_broad.get(detailed_classes[best_idx], ""),
            "confidence": float(probs_full[best_idx]),
            "fallback": True,
        }

    # Mask: set invalid indices to -inf
    masked = np.full_like(detailed_logits_row, -np.inf)
    for idx in valid_indices:
        masked[idx] = detailed_logits_row[idx]

    # Softmax over valid entries only
    probs = softmax(masked.reshape(1, -1))[0]
    best_idx = int(np.argmax(probs))
    best_detailed = detailed_classes[best_idx]

    return {
        "predicted_detailed_field": best_detailed,
        "predicted_major_field": major_name,
        "predicted_broad_field": detailed_to_broad.get(best_detailed, ""),
        "confidence": float(probs[best_idx]),
        "fallback": False,
    }


def evaluate_strategy(
    strategy_name: str,
    predictions: list[dict],
    records: list[dict],
    detailed_to_major: dict[str, str],
) -> dict:
    """Compute metrics for a strategy's predictions."""
    n = len(predictions)
    assert n == len(records)

    detailed_correct = 0
    major_correct = 0
    broad_correct = 0

    tp_d: Counter = Counter()
    fp_d: Counter = Counter()
    fn_d: Counter = Counter()

    for pred, rec in zip(predictions, records):
        true_detailed = rec.get("detailed_field", "")
        true_major = rec.get("major_field", "")
        true_broad = rec.get("broad_field", "")

        pred_detailed = pred["predicted_detailed_field"]
        pred_major = pred["predicted_major_field"]
        pred_broad = pred["predicted_broad_field"]

        if pred_detailed == true_detailed:
            detailed_correct += 1
            tp_d[true_detailed] += 1
        else:
            fp_d[pred_detailed] += 1
            fn_d[true_detailed] += 1

        if pred_major == true_major:
            major_correct += 1
        if pred_broad == true_broad:
            broad_correct += 1

    detailed_acc = detailed_correct / n
    major_acc = major_correct / n
    broad_acc = broad_correct / n

    # Macro F1 (detailed)
    all_fields = set(tp_d.keys()) | set(fp_d.keys()) | set(fn_d.keys())
    f1_values = []
    for fld in all_fields:
        prec = tp_d[fld] / (tp_d[fld] + fp_d[fld]) if (tp_d[fld] + fp_d[fld]) > 0 else 0
        rec_val = tp_d[fld] / (tp_d[fld] + fn_d[fld]) if (tp_d[fld] + fn_d[fld]) > 0 else 0
        f1 = 2 * prec * rec_val / (prec + rec_val) if (prec + rec_val) > 0 else 0
        if (tp_d[fld] + fn_d[fld]) > 0:
            f1_values.append(f1)
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0

    return {
        "strategy": strategy_name,
        "detailed_field_accuracy": round(detailed_acc, 4),
        "detailed_macro_f1": round(macro_f1, 4),
        "major_field_accuracy": round(major_acc, 4),
        "broad_field_accuracy": round(broad_acc, 4),
        "n_eval": n,
        "detailed_correct": detailed_correct,
        "major_correct": major_correct,
    }


def main():
    parser = argparse.ArgumentParser(description="Hierarchical inference via constrained decoding")
    parser.add_argument("--major-model", type=str, default=None,
                        help="Path to major-field model directory")
    parser.add_argument("--detailed-model", type=str, default=None,
                        help="Path to detailed-field model directory")
    parser.add_argument("--test-data", type=str, default="data/generated/test.jsonl",
                        help="Test data JSONL")
    parser.add_argument("--output-dir", type=str, default="output/predictions",
                        help="Output directory")
    parser.add_argument("--top-k-major", type=int, default=5,
                        help="Number of top major fields to consider for strategies B/C/D")
    parser.add_argument("--major-alpha", type=float, default=2.0,
                        help="Exponent for major prob in strategy D (higher = trust major more)")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["top1_mask", "topk_max", "combined", "weighted"],
                        help="Run only one strategy (default: all four)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Inference batch size")
    args = parser.parse_args()

    project_root = find_project_root()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model paths
    if args.major_model:
        major_model_dir = Path(args.major_model)
    else:
        # Auto-find: scan sweep models for any with label_classes.json,
        # prefer the one with most classes (should be 74 for major-field)
        sweep_dir = project_root / "output" / "sweep" / "models"
        finetune_dir = project_root / "output" / "models" / "finetune"
        found = []
        if sweep_dir.is_dir():
            for d in sorted(sweep_dir.iterdir()):
                lc = d / "label_classes.json"
                if lc.exists():
                    with open(lc) as f:
                        n = len(json.load(f))
                    found.append((d, n))
        if finetune_dir.is_dir() and (finetune_dir / "label_classes.json").exists():
            with open(finetune_dir / "label_classes.json") as f:
                n = len(json.load(f))
            found.append((finetune_dir, n))

        # Pick the one with exactly 74 classes (major-field), or closest
        major_candidates = [d for d, n in found if 60 <= n <= 80]
        if major_candidates:
            major_model_dir = major_candidates[0]
            print(f"Auto-discovered major model: {major_model_dir.name}")
        elif found:
            major_model_dir = found[0][0]
            print(f"Auto-discovered model (best guess): {major_model_dir.name}")
        else:
            major_model_dir = None

        if major_model_dir is None:
            print("ERROR: Could not find major-field model. Use --major-model to specify.")
            return

    if args.detailed_model:
        detailed_model_dir = Path(args.detailed_model)
    else:
        detailed_model_dir = project_root / "output" / "models" / "detailed_finetune"

    print(f"Major model:    {major_model_dir}")
    print(f"Detailed model: {detailed_model_dir}")

    # Load taxonomy
    taxonomy_path = project_root / "data" / "processed" / "SEDCIP24.json"
    detailed_to_major, detailed_to_broad, major_to_detailed_set = build_taxonomy_mappings(taxonomy_path)

    # Load test data
    test_path = project_root / args.test_data
    records = load_jsonl(test_path)
    test_texts = [r["abstract"] for r in records]
    print(f"Test records:   {len(records)}")
    print()

    # Run both models
    print("Running major-field model...")
    major_logits, major_classes = load_model_and_predict(
        major_model_dir, test_texts, batch_size=args.batch_size,
    )
    print(f"  {len(major_classes)} major classes, {major_logits.shape[0]} predictions")

    print("Running detailed-field model...")
    detailed_logits, detailed_classes = load_model_and_predict(
        detailed_model_dir, test_texts, batch_size=args.batch_size,
    )
    print(f"  {len(detailed_classes)} detailed classes, {detailed_logits.shape[0]} predictions")

    # Build index mapping: for each major class index, which detailed indices belong to it
    major_to_detailed_idx = build_major_to_detailed_indices(
        major_classes, detailed_classes, major_to_detailed_set,
    )

    # Check coverage
    covered = sum(1 for indices in major_to_detailed_idx.values() if indices)
    print(f"\n  Major→detailed mapping coverage: {covered}/{len(major_classes)} majors have mapped detailed fields")
    unmapped = [major_classes[i] for i, indices in major_to_detailed_idx.items() if not indices]
    if unmapped:
        print(f"  WARNING: {len(unmapped)} major fields have no mapped detailed fields:")
        for m in unmapped[:10]:
            print(f"    - {m}")

    # Run hierarchical prediction
    print(f"\nRunning hierarchical inference (top_k_major={args.top_k_major}, alpha={args.major_alpha})...")
    all_results = hierarchical_predict(
        major_logits, detailed_logits,
        major_classes, detailed_classes,
        major_to_detailed_idx,
        detailed_to_major, detailed_to_broad,
        top_k_major=args.top_k_major,
        major_alpha=args.major_alpha,
    )

    # Also compute flat baseline for comparison
    print("\nComputing flat detailed baseline...")
    flat_probs = softmax(detailed_logits)
    flat_preds = []
    for i in range(len(records)):
        best_idx = int(flat_probs[i].argmax())
        best_detailed = detailed_classes[best_idx]
        flat_preds.append({
            "predicted_detailed_field": best_detailed,
            "predicted_major_field": detailed_to_major.get(best_detailed, ""),
            "predicted_broad_field": detailed_to_broad.get(best_detailed, ""),
            "confidence": float(flat_probs[i][best_idx]),
        })

    # Evaluate all strategies
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    strategies_to_eval = ["top1_mask", "topk_max", "combined", "weighted"] if args.strategy is None else [args.strategy]

    all_metrics = []

    # Flat baseline
    flat_metrics = evaluate_strategy("flat_detailed", flat_preds, records, detailed_to_major)
    all_metrics.append(flat_metrics)
    print(f"\n  {'Flat (no hierarchy)':<25s}  "
          f"detailed={flat_metrics['detailed_field_accuracy']:.4f}  "
          f"macro_f1={flat_metrics['detailed_macro_f1']:.4f}  "
          f"major={flat_metrics['major_field_accuracy']:.4f}  "
          f"broad={flat_metrics['broad_field_accuracy']:.4f}")

    # Hierarchical strategies
    for strategy in strategies_to_eval:
        metrics = evaluate_strategy(f"hierarchical_{strategy}", all_results[strategy], records, detailed_to_major)
        all_metrics.append(metrics)

        label = {"top1_mask": "A) Top-1 mask", "topk_max": "B) Top-k max", "combined": "C) Combined", "weighted": f"D) Weighted (α={args.major_alpha})"}[strategy]
        print(f"  {label:<25s}  "
              f"detailed={metrics['detailed_field_accuracy']:.4f}  "
              f"macro_f1={metrics['detailed_macro_f1']:.4f}  "
              f"major={metrics['major_field_accuracy']:.4f}  "
              f"broad={metrics['broad_field_accuracy']:.4f}")

    # Improvement summary
    best_hier = max(all_metrics[1:], key=lambda m: m["detailed_field_accuracy"])
    delta = best_hier["detailed_field_accuracy"] - flat_metrics["detailed_field_accuracy"]
    print(f"\n  Best hierarchical improvement over flat: {delta:+.4f} "
          f"({flat_metrics['detailed_field_accuracy']:.4f} → {best_hier['detailed_field_accuracy']:.4f})")
    print(f"  Best strategy: {best_hier['strategy']}")

    # Save results
    results_file = output_dir / "metrics_hierarchical_comparison.json"
    with open(results_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Metrics saved: {results_file}")

    # Save best strategy predictions
    best_strategy_name = best_hier["strategy"].replace("hierarchical_", "")
    best_preds = all_results[best_strategy_name]

    pred_records = []
    for i, (pred, rec) in enumerate(zip(best_preds, records)):
        pred_records.append({
            "abstract": rec["abstract"],
            "true_detailed_field": rec.get("detailed_field", ""),
            "true_major_field": rec.get("major_field", ""),
            "true_broad_field": rec.get("broad_field", ""),
            "predicted_detailed_field": pred["predicted_detailed_field"],
            "predicted_major_field": pred["predicted_major_field"],
            "predicted_broad_field": pred["predicted_broad_field"],
            "confidence": pred["confidence"],
            "major_confidence": pred.get("major_confidence", 0),
            "strategy": best_strategy_name,
        })

    pred_file = output_dir / f"predictions_hierarchical_{best_strategy_name}_synthetic_test.json"
    with open(pred_file, "w") as f:
        json.dump({
            "model_name": f"hierarchical_{best_strategy_name}",
            "dataset": "synthetic_test",
            "metadata": {
                "major_model": str(major_model_dir),
                "detailed_model": str(detailed_model_dir),
                "strategy": best_strategy_name,
                "top_k_major": args.top_k_major,
                "n_major_classes": len(major_classes),
                "n_detailed_classes": len(detailed_classes),
            },
            "predictions": pred_records,
        }, f, indent=2)
    print(f"  Predictions saved: {pred_file}")

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
