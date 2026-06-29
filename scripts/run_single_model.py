"""D2.3: Train single unified 315-class model + marginalization-based hierarchical inference.

Trains one SciBERT model on enriched detailed-field data (original synthetic +
targeted synthetic + detailed silver labels), then evaluates using:
  1. Flat detailed predictions
  2. Marginalized major predictions (sum detailed probs per major)
  3. Strategy C with marginalized major × detailed probs

Outputs go to output/models/single_model/ and output/predictions/ with
d2_ prefix to avoid overwriting C4/C5 results.

Usage:
    # Train + evaluate (GPU required):
    python scripts/run_single_model.py

    # Train on enriched data:
    python scripts/run_single_model.py --train-data data/generated/train_d2.jsonl

    # Predict-only from saved model:
    python scripts/run_single_model.py --predict-only

    # Compare with two-model approach:
    python scripts/run_single_model.py --predict-only --compare
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def build_taxonomy_mappings(taxonomy_path: Path) -> tuple[dict, dict, dict]:
    """Build detailed→major, detailed→broad, and major→set(detailed) mappings."""
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)

    detailed_to_major = {}
    detailed_to_broad = {}
    major_to_detailed: dict[str, set] = {}

    for entry in taxonomy:
        d = entry["Detailed_Field_label"]
        m = entry["Major_Field_label"]
        b = entry["Broad_Field_label"]
        detailed_to_major.setdefault(d, m)
        detailed_to_broad.setdefault(d, b)
        major_to_detailed.setdefault(m, set()).add(d)

    return detailed_to_major, detailed_to_broad, major_to_detailed


def log_class_distribution(labels: list[str], label_type: str):
    counts = Counter(labels)
    n_classes = len(counts)
    min_count = min(counts.values())
    max_count = max(counts.values())
    n_rare = sum(1 for c in counts.values() if c < 10)
    print(f"\n{label_type}: {n_classes} classes, {len(labels)} samples")
    print(f"  Range: {min_count}–{max_count} per class")
    if n_rare:
        print(f"  WARNING: {n_rare} classes have <10 samples")


def train_single_model(
    project_root: Path,
    train_path: Path,
    model_name: str = "allenai/scibert_scivocab_uncased",
    lr: float = 3e-5,
    epochs: int = 8,
    batch_size: int = 16,
    freeze_layers: int = 8,
    seed: int = 42,
    max_length: int = 512,
    output_model_dir: Path | None = None,
):
    """Train SciBERT on detailed_field labels with layer freezing."""
    import torch
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    # Load training data — filter to records with detailed_field
    records = load_jsonl(train_path)
    valid = [r for r in records if r.get("detailed_field")]
    skipped = len(records) - len(valid)
    if skipped:
        print(f"WARNING: {skipped}/{len(records)} records missing detailed_field, skipping them")
        # Show sources of skipped records
        skip_sources = Counter(r.get("source", "unknown") for r in records if not r.get("detailed_field"))
        for src, n in skip_sources.most_common():
            print(f"  Skipped from source '{src}': {n}")

    texts = [r["abstract"] for r in valid]
    labels = [r["detailed_field"] for r in valid]
    sources = Counter(r.get("source", "unknown") for r in valid)
    print(f"\nTraining data by source:")
    for src, n in sources.most_common():
        print(f"  {src}: {n}")

    log_class_distribution(labels, "detailed_field (train)")

    # Encode labels
    le = LabelEncoder()
    le.fit(labels)
    y = le.transform(labels).tolist()
    n_classes = len(le.classes_)
    label_classes = list(le.classes_)
    print(f"Number of detailed field classes: {n_classes}")

    # 90/10 stratified split
    idx_train, idx_val = train_test_split(
        list(range(len(texts))), test_size=0.1, random_state=seed, stratify=y,
    )
    tr_texts = [texts[i] for i in idx_train]
    tr_labels = [y[i] for i in idx_train]
    val_texts = [texts[i] for i in idx_val]
    val_labels = [y[i] for i in idx_val]
    print(f"Train: {len(tr_texts)}, Val: {len(val_texts)}")

    # Load model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)

    # Apply layer freezing
    if freeze_layers > 0:
        encoder = None
        if hasattr(model, "bert"):
            encoder = model.bert.encoder
        elif hasattr(model, "deberta"):
            encoder = model.deberta.encoder
        if encoder and hasattr(encoder, "layer"):
            for i, layer in enumerate(encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(f"  Froze first {freeze_layers} encoder layers")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")

    train_ds = Dataset.from_dict({"text": tr_texts, "label": tr_labels}).map(tokenize_fn, batched=True)
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(tokenize_fn, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    if output_model_dir is None:
        output_model_dir = project_root / "output" / "models" / "single_model"
    output_model_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_model_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        seed=seed,
        report_to="none",
    )

    def compute_hf_metrics(eval_pred):
        logits, lab = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == lab).mean())}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_hf_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"\nTraining {model_name} for {epochs} epochs (lr={lr}, bs={batch_size}, freeze={freeze_layers})...")
    trainer.train()
    print("Training complete.\n")

    # Save
    trainer.save_model(str(output_model_dir))
    tokenizer.save_pretrained(str(output_model_dir))
    with open(output_model_dir / "label_classes.json", "w") as f:
        json.dump(label_classes, f, indent=2)
    print(f"Model saved to: {output_model_dir}")

    return trainer, tokenizer, label_classes


def marginalize_to_major(
    detailed_probs: np.ndarray,
    detailed_classes: list[str],
    detailed_to_major: dict[str, str],
) -> tuple[np.ndarray, list[str]]:
    """Sum detailed probabilities per major field to get marginalized major probs.

    Returns (major_probs [n x n_major], major_classes list).
    """
    # Build unique major field list (preserving taxonomy order)
    seen = set()
    major_classes = []
    for d in detailed_classes:
        m = detailed_to_major.get(d, "")
        if m and m not in seen:
            seen.add(m)
            major_classes.append(m)

    # Build mapping: major_idx -> list of detailed_idx
    major_to_detail_idx: dict[int, list[int]] = {}
    major_name_to_idx = {m: i for i, m in enumerate(major_classes)}
    for di, d in enumerate(detailed_classes):
        m = detailed_to_major.get(d, "")
        mi = major_name_to_idx.get(m)
        if mi is not None:
            major_to_detail_idx.setdefault(mi, []).append(di)

    # Marginalize: P(major) = sum P(detailed) for all detailed in major
    n = detailed_probs.shape[0]
    major_probs = np.zeros((n, len(major_classes)), dtype=np.float64)
    for mi, detail_indices in major_to_detail_idx.items():
        major_probs[:, mi] = detailed_probs[:, detail_indices].sum(axis=1)

    return major_probs, major_classes


def evaluate_single_model(
    project_root: Path,
    model_dir: Path,
    test_path: Path,
    output_dir: Path,
    batch_size: int = 32,
    max_length: int = 512,
    compare: bool = False,
):
    """Evaluate single model with flat + marginalized + Strategy C approaches."""
    import torch
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    taxonomy_path = project_root / "data" / "processed" / "SEDCIP24.json"
    detailed_to_major, detailed_to_broad, major_to_detailed = build_taxonomy_mappings(taxonomy_path)

    # Load model
    with open(model_dir / "label_classes.json") as f:
        label_classes = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # Load test data
    records = load_jsonl(test_path)
    valid = [r for r in records if r.get("detailed_field")]
    texts = [r["abstract"] for r in valid]
    true_detailed = [r["detailed_field"] for r in valid]
    true_major = [detailed_to_major.get(r["detailed_field"], r.get("major_field", "")) for r in valid]
    true_broad = [detailed_to_broad.get(r["detailed_field"], r.get("broad_field", "")) for r in valid]
    print(f"Evaluating on {len(texts)} test abstracts...")

    # Get logits
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")

    ds = Dataset.from_dict({"text": texts}).map(tokenize_fn, batched=True)
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
    detailed_probs = softmax(logits)

    # === Strategy 1: Flat detailed predictions ===
    flat_pred_idx = np.argmax(detailed_probs, axis=1)
    flat_pred_detailed = [label_classes[i] for i in flat_pred_idx]
    flat_pred_major = [detailed_to_major.get(d, "") for d in flat_pred_detailed]
    flat_pred_broad = [detailed_to_broad.get(d, "") for d in flat_pred_detailed]
    flat_conf = [float(detailed_probs[i, flat_pred_idx[i]]) for i in range(len(texts))]

    flat_detailed_acc = accuracy_score(true_detailed, flat_pred_detailed)
    flat_major_acc = accuracy_score(true_major, flat_pred_major)
    flat_broad_acc = accuracy_score(true_broad, flat_pred_broad)
    flat_macro_f1 = f1_score(true_detailed, flat_pred_detailed, average="macro", zero_division=0)
    flat_major_f1 = f1_score(true_major, flat_pred_major, average="macro", zero_division=0)

    print(f"\n=== Flat Detailed (single model) ===")
    print(f"  Detailed acc: {flat_detailed_acc:.4f}")
    print(f"  Major acc (rolled up): {flat_major_acc:.4f}")
    print(f"  Broad acc (rolled up): {flat_broad_acc:.4f}")
    print(f"  Detailed macro F1: {flat_macro_f1:.4f}")
    print(f"  Major macro F1: {flat_major_f1:.4f}")

    # === Strategy 2: Marginalized major predictions ===
    major_probs, major_classes = marginalize_to_major(detailed_probs, label_classes, detailed_to_major)
    marg_major_idx = np.argmax(major_probs, axis=1)
    marg_pred_major = [major_classes[i] for i in marg_major_idx]

    marg_major_acc = accuracy_score(true_major, marg_pred_major)
    marg_major_f1 = f1_score(true_major, marg_pred_major, average="macro", zero_division=0)

    print(f"\n=== Marginalized Major (single model) ===")
    print(f"  Major acc: {marg_major_acc:.4f}")
    print(f"  Major macro F1: {marg_major_f1:.4f}")

    # === Strategy 3: Strategy C with marginalized major × detailed ===
    # Build major_to_detailed_indices
    major_name_to_idx = {m: i for i, m in enumerate(major_classes)}
    major_to_detail_idx: dict[int, list[int]] = {}
    for di, d in enumerate(label_classes):
        m = detailed_to_major.get(d, "")
        mi = major_name_to_idx.get(m)
        if mi is not None:
            major_to_detail_idx.setdefault(mi, []).append(di)

    strat_c_detailed = []
    strat_c_major = []
    strat_c_broad = []
    strat_c_conf = []
    top_k = 5

    for i in range(len(texts)):
        top_major_idx = np.argsort(major_probs[i])[::-1][:top_k]
        best_score = -1.0
        best_d = ""
        best_m = ""
        best_b = ""

        for mi in top_major_idx:
            m_prob = major_probs[i, mi]
            valid_d = major_to_detail_idx.get(int(mi), [])
            for di in valid_d:
                score = float(m_prob * detailed_probs[i, di])
                if score > best_score:
                    best_score = score
                    best_d = label_classes[di]
                    best_m = major_classes[mi]
                    best_b = detailed_to_broad.get(label_classes[di], "")

        strat_c_detailed.append(best_d)
        strat_c_major.append(best_m)
        strat_c_broad.append(best_b)
        strat_c_conf.append(best_score)

    sc_detailed_acc = accuracy_score(true_detailed, strat_c_detailed)
    sc_major_acc = accuracy_score(true_major, strat_c_major)
    sc_broad_acc = accuracy_score(true_broad, strat_c_broad)
    sc_macro_f1 = f1_score(true_detailed, strat_c_detailed, average="macro", zero_division=0)

    print(f"\n=== Strategy C Marginalized (single model) ===")
    print(f"  Detailed acc: {sc_detailed_acc:.4f}")
    print(f"  Major acc (rolled up): {sc_major_acc:.4f}")
    print(f"  Broad acc (rolled up): {sc_broad_acc:.4f}")
    print(f"  Detailed macro F1: {sc_macro_f1:.4f}")

    # === Save results ===
    output_dir.mkdir(parents=True, exist_ok=True)

    # Metrics comparison
    results = {
        "single_model_flat": {
            "detailed_accuracy": round(flat_detailed_acc, 4),
            "major_accuracy": round(flat_major_acc, 4),
            "broad_accuracy": round(flat_broad_acc, 4),
            "detailed_macro_f1": round(flat_macro_f1, 4),
            "major_macro_f1": round(flat_major_f1, 4),
        },
        "single_model_marginalized_major": {
            "major_accuracy": round(marg_major_acc, 4),
            "major_macro_f1": round(marg_major_f1, 4),
        },
        "single_model_strategy_c": {
            "detailed_accuracy": round(sc_detailed_acc, 4),
            "major_accuracy": round(sc_major_acc, 4),
            "broad_accuracy": round(sc_broad_acc, 4),
            "detailed_macro_f1": round(sc_macro_f1, 4),
        },
    }

    # Add two-model comparison if available
    two_model_path = output_dir / "metrics_hierarchical_comparison.json"
    if compare and two_model_path.exists():
        with open(two_model_path) as f:
            two_model = json.load(f)
        for entry in two_model:
            if entry["strategy"] == "hierarchical_combined":
                results["two_model_strategy_c_reference"] = {
                    "detailed_accuracy": entry["detailed_field_accuracy"],
                    "major_accuracy": entry["major_field_accuracy"],
                    "broad_accuracy": entry["broad_field_accuracy"],
                    "detailed_macro_f1": entry["detailed_macro_f1"],
                }
                break
        # Compute deltas
        ref = results.get("two_model_strategy_c_reference", {})
        if ref:
            sc = results["single_model_strategy_c"]
            results["delta_vs_two_model"] = {
                "detailed_accuracy": round(sc["detailed_accuracy"] - ref["detailed_accuracy"], 4),
                "major_accuracy": round(sc["major_accuracy"] - ref["major_accuracy"], 4),
                "detailed_macro_f1": round(sc["detailed_macro_f1"] - ref["detailed_macro_f1"], 4),
            }

    metrics_path = output_dir / "metrics_d2_single_model_comparison.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Save predictions (Strategy C — the best)
    predictions = {
        "model_name": "single_model_strategy_c",
        "dataset": "synthetic_test",
        "metadata": {
            "model_dir": str(model_dir),
            "strategy": "marginalized_combined",
            "n_detailed_classes": len(label_classes),
            "n_major_classes_marginalized": len(major_classes),
            "train_data": str(test_path),
        },
        "predictions": [],
    }
    for i in range(len(texts)):
        predictions["predictions"].append({
            "abstract": texts[i],
            "true_detailed_field": true_detailed[i],
            "true_major_field": true_major[i],
            "true_broad_field": true_broad[i],
            "predicted_detailed_field": strat_c_detailed[i],
            "predicted_major_field": strat_c_major[i],
            "predicted_broad_field": strat_c_broad[i],
            "confidence": round(strat_c_conf[i], 6),
            "strategy": "marginalized_combined",
        })

    pred_path = output_dir / "predictions_d2_single_model_synthetic_test.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to: {pred_path}")

    # === Per-field F1 chart ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        chart_dir = project_root / "output" / "charts"
        chart_dir.mkdir(parents=True, exist_ok=True)

        # Major-field per-field F1 comparison
        report = classification_report(true_major, strat_c_major, output_dict=True, zero_division=0)
        fields_f1 = {k: v["f1-score"] for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg")}
        sorted_fields = sorted(fields_f1.items(), key=lambda x: x[1], reverse=True)

        fig, ax = plt.subplots(figsize=(10, max(8, len(sorted_fields) * 0.22)))
        names = [f[0][:50] for f in sorted_fields]
        scores = [f[1] for f in sorted_fields]
        colors = ["#4caf50" if s >= 0.8 else "#ff9800" if s >= 0.5 else "#f44336" for s in scores]
        ax.barh(range(len(names)), scores, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("F1 Score")
        ax.set_title("Per-Field F1 Score — Single Model (D2)")
        ax.axvline(x=0.8, color="gray", linestyle="--", alpha=0.5)
        ax.invert_yaxis()
        plt.tight_layout()
        chart_path = chart_dir / "d2_per_field_f1_single_model.png"
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Chart saved to: {chart_path}")

        # Comparison bar chart: single vs two-model
        if compare and "two_model_strategy_c_reference" in results:
            ref = results["two_model_strategy_c_reference"]
            sc = results["single_model_strategy_c"]
            fig, ax = plt.subplots(figsize=(8, 5))
            metrics_names = ["Detailed Acc", "Major Acc", "Broad Acc", "Detailed F1"]
            two_vals = [ref["detailed_accuracy"], ref["major_accuracy"], ref.get("broad_accuracy", 0), ref["detailed_macro_f1"]]
            one_vals = [sc["detailed_accuracy"], sc["major_accuracy"], sc.get("broad_accuracy", 0), sc["detailed_macro_f1"]]
            x = np.arange(len(metrics_names))
            w = 0.35
            ax.bar(x - w/2, [v*100 for v in two_vals], w, label="Two-model (C5b)", color="#93b5f1")
            ax.bar(x + w/2, [v*100 for v in one_vals], w, label="Single-model (D2)", color="#1a5fb4")
            ax.set_ylabel("Score (%)")
            ax.set_title("Single Model vs Two-Model Comparison")
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_names)
            ax.legend()
            ax.set_ylim(60, 100)
            for i, (tv, sv) in enumerate(zip(two_vals, one_vals)):
                delta = (sv - tv) * 100
                color = "#2e7d32" if delta >= 0 else "#c62828"
                ax.annotate(f"{delta:+.1f}%", xy=(i + w/2, sv*100 + 0.5), ha="center", fontsize=9, color=color, fontweight="bold")
            plt.tight_layout()
            chart_path = chart_dir / "d2_single_vs_two_model.png"
            fig.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Chart saved to: {chart_path}")
    except ImportError:
        print("matplotlib not available, skipping charts")

    # Print comparison summary
    if compare and "delta_vs_two_model" in results:
        print(f"\n=== Single-Model vs Two-Model Comparison ===")
        d = results["delta_vs_two_model"]
        for k, v in d.items():
            sign = "+" if v >= 0 else ""
            print(f"  {k}: {sign}{v:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="D2.3: Train + evaluate single unified model")
    parser.add_argument("--predict-only", action="store_true", help="Load saved model and evaluate only")
    parser.add_argument("--model-dir", type=Path, default=None, help="Model directory (default: output/models/single_model)")
    parser.add_argument("--train-data", type=Path, default=None, help="Training data JSONL")
    parser.add_argument("--test-data", type=Path, default=None, help="Test data JSONL")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for predictions/metrics")
    parser.add_argument("--model-name", type=str, default="allenai/scibert_scivocab_uncased", help="HF model name")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--freeze-layers", type=int, default=8, help="Freeze first N encoder layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compare", action="store_true", help="Compare with two-model approach")
    args = parser.parse_args()

    project_root = find_project_root()
    model_dir = args.model_dir or (project_root / "output" / "models" / "single_model")
    test_path = args.test_data or (project_root / "data" / "generated" / "test.jsonl")
    output_dir = args.output_dir or (project_root / "output" / "predictions")

    if args.predict_only:
        if not (model_dir / "label_classes.json").exists():
            print(f"ERROR: No saved model found at {model_dir}")
            sys.exit(1)
        evaluate_single_model(project_root, model_dir, test_path, output_dir, compare=args.compare)
    else:
        train_path = args.train_data
        if train_path is None:
            # Prefer enriched data, fall back to original
            d2_path = project_root / "data" / "generated" / "train_d2.jsonl"
            if d2_path.exists():
                train_path = d2_path
                print(f"Using enriched training data: {train_path}")
            else:
                train_path = project_root / "data" / "generated" / "train.jsonl"
                print(f"Using original training data: {train_path}")
                print("  (Run generate_targeted.py --merge first for enriched data)")

        train_single_model(
            project_root=project_root,
            train_path=train_path,
            model_name=args.model_name,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            freeze_layers=args.freeze_layers,
            seed=args.seed,
            output_model_dir=model_dir,
        )

        evaluate_single_model(project_root, model_dir, test_path, output_dir, compare=args.compare)


if __name__ == "__main__":
    main()
