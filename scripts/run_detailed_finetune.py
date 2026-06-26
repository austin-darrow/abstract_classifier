"""Train and evaluate SciBERT on detailed CIP fields (~315-418 classes).

Standalone script — does NOT modify the existing major-field pipeline.
Uses the same best hyperparameters as the major-field model (C4 winner):
  SciBERT, lr=3e-5, 8 epochs, bs=16, seed=42

Usage:
    # Train + evaluate (GPU required):
    python scripts/run_detailed_finetune.py

    # Predict-only from saved model (GPU required for inference):
    python scripts/run_detailed_finetune.py --predict-only

    # Custom hyperparams:
    python scripts/run_detailed_finetune.py --epochs 5 --lr 2e-5
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


def build_taxonomy_mappings(taxonomy_path: Path) -> tuple[dict, dict]:
    """Build detailed→major and detailed→broad mappings from SEDCIP24.json.

    Returns:
        (detailed_to_major, detailed_to_broad) dicts
    """
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)

    detailed_to_major = {}
    detailed_to_broad = {}
    for entry in taxonomy:
        d = entry["Detailed_Field_label"]
        detailed_to_major.setdefault(d, entry["Major_Field_label"])
        detailed_to_broad.setdefault(d, entry["Broad_Field_label"])
    return detailed_to_major, detailed_to_broad


def log_class_distribution(labels: list[str], label_type: str = "detailed_field"):
    """Log distribution stats and flag low-support classes."""
    counts = Counter(labels)
    n_classes = len(counts)
    values = sorted(counts.values())
    total = sum(values)

    print(f"\n{'='*60}")
    print(f"Class Distribution ({label_type})")
    print(f"{'='*60}")
    print(f"  Total samples: {total:,}")
    print(f"  Unique classes: {n_classes}")
    print(f"  Min samples/class: {values[0]}")
    print(f"  Max samples/class: {values[-1]}")
    print(f"  Median samples/class: {values[len(values)//2]}")
    print(f"  Mean samples/class: {total/n_classes:.1f}")

    # Flag low-support classes
    low_support = [(f, c) for f, c in counts.items() if c < 10]
    if low_support:
        low_support.sort(key=lambda x: x[1])
        print(f"\n  WARNING: {len(low_support)} classes with <10 training samples:")
        for field, count in low_support[:20]:
            print(f"    {count:3d}  {field}")
        if len(low_support) > 20:
            print(f"    ... and {len(low_support) - 20} more")
    print()


def train_detailed_model(
    project_root: Path,
    train_path: Path,
    model_name: str = "allenai/scibert_scivocab_uncased",
    lr: float = 3e-5,
    epochs: int = 8,
    batch_size: int = 16,
    seed: int = 42,
    max_length: int = 512,
    output_model_dir: Path | None = None,
) -> tuple:
    """Train SciBERT on detailed_field labels.

    Returns:
        (trainer, tokenizer, label_classes, detailed_to_major, detailed_to_broad)
    """
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

    taxonomy_path = project_root / "data" / "processed" / "SEDCIP24.json"
    detailed_to_major, detailed_to_broad = build_taxonomy_mappings(taxonomy_path)

    # Load training data
    records = load_jsonl(train_path)
    texts = [r["abstract"] for r in records]
    labels = [r["detailed_field"] for r in records]

    # Check for records missing detailed_field
    missing = sum(1 for r in records if not r.get("detailed_field"))
    if missing:
        print(f"WARNING: {missing}/{len(records)} records missing detailed_field, skipping them")
        texts = [r["abstract"] for r in records if r.get("detailed_field")]
        labels = [r["detailed_field"] for r in records if r.get("detailed_field")]

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

    # Tokenize
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")

    train_ds = Dataset.from_dict({"text": tr_texts, "label": tr_labels}).map(tokenize_fn, batched=True)
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(tokenize_fn, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Model output directory
    if output_model_dir is None:
        output_model_dir = project_root / "output" / "models" / "detailed_finetune"
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

    print(f"\nTraining {model_name} for {epochs} epochs (lr={lr}, bs={batch_size})...")
    trainer.train()
    print("Training complete.\n")

    # Save model, tokenizer, and label classes
    trainer.save_model(str(output_model_dir))
    tokenizer.save_pretrained(str(output_model_dir))
    with open(output_model_dir / "label_classes.json", "w") as f:
        json.dump(label_classes, f, indent=2)
    print(f"Model saved to: {output_model_dir}")

    return trainer, tokenizer, label_classes, detailed_to_major, detailed_to_broad


def predict_and_evaluate(
    project_root: Path,
    test_path: Path,
    trainer_or_model,
    tokenizer,
    label_classes: list[str],
    detailed_to_major: dict,
    detailed_to_broad: dict,
    output_dir: Path,
    dataset_name: str = "synthetic_test",
    batch_size: int = 32,
    max_length: int = 512,
    model_label: str = "detailed_scibert_8ep",
    use_trainer: bool = True,
):
    """Predict on test set and compute metrics at all taxonomy levels."""
    import torch
    from datasets import Dataset

    # Load test data
    records = load_jsonl(test_path)
    print(f"Loaded {len(records)} test records from {test_path.name}")

    test_texts = [r["abstract"] for r in records]
    true_detailed = [r.get("detailed_field", "") for r in records]
    true_major = [r.get("major_field", "") for r in records]
    true_broad = [r.get("broad_field", "") for r in records]

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")

    test_ds = Dataset.from_dict({"text": test_texts, "label": [0] * len(test_texts)}).map(tokenize_fn, batched=True)
    test_ds.set_format("torch", columns=["input_ids", "attention_mask"])

    # Get predictions
    if use_trainer:
        raw = trainer_or_model.predict(test_ds)
        logits = raw.predictions
    else:
        model = trainer_or_model
        model.eval()
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(test_ds), batch_size):
                batch = test_ds[i:i + batch_size]
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(outputs.logits.cpu().numpy())
        logits = np.concatenate(all_logits, axis=0)

    probs = _softmax(logits)
    y_pred = probs.argmax(axis=1)

    # Build prediction records
    predictions = []
    for i in range(len(records)):
        pred_detailed = label_classes[y_pred[i]]
        pred_major = detailed_to_major.get(pred_detailed, "")
        pred_broad = detailed_to_broad.get(pred_detailed, "")
        confidence = float(probs[i].max())

        top_k_idx = np.argsort(probs[i])[::-1][:10]
        top_k_detailed = [label_classes[idx] for idx in top_k_idx]
        top_k_scores = [float(probs[i][idx]) for idx in top_k_idx]
        top_k_major = [detailed_to_major.get(d, "") for d in top_k_detailed]

        predictions.append({
            "abstract": records[i]["abstract"],
            "true_detailed_field": true_detailed[i],
            "true_major_field": true_major[i],
            "true_broad_field": true_broad[i],
            "predicted_detailed_field": pred_detailed,
            "predicted_major_field": pred_major,
            "predicted_broad_field": pred_broad,
            "confidence": confidence,
            "top_k_detailed_fields": top_k_detailed,
            "top_k_major_fields": top_k_major,
            "top_k_scores": top_k_scores,
        })

    # Compute metrics
    metrics = compute_detailed_metrics(predictions, label_classes, detailed_to_major)

    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / f"predictions_{model_label}_{dataset_name}.json"
    with open(pred_file, "w") as f:
        json.dump({
            "model_name": model_label,
            "dataset": dataset_name,
            "metadata": {
                "n_classes": len(label_classes),
                "n_test": len(records),
                "taxonomy_level": "detailed_field",
            },
            "predictions": predictions,
        }, f, indent=2)
    print(f"Predictions saved: {pred_file}")

    # Save metrics
    metrics_file = output_dir / f"metrics_{model_label}_{dataset_name}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_file}")

    return predictions, metrics


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def compute_detailed_metrics(
    predictions: list[dict],
    label_classes: list[str],
    detailed_to_major: dict,
) -> dict:
    """Compute metrics at detailed, major, and broad field levels."""
    n = len(predictions)

    # Filter to records with ground truth
    has_truth = [p for p in predictions if p["true_detailed_field"]]
    n_eval = len(has_truth)
    if n_eval == 0:
        print("WARNING: No records with ground truth detailed_field labels")
        return {}

    print(f"\n{'='*60}")
    print(f"Evaluation Results ({n_eval} records with ground truth)")
    print(f"{'='*60}")

    # --- Detailed field metrics ---
    detailed_correct = sum(
        1 for p in has_truth
        if p["predicted_detailed_field"] == p["true_detailed_field"]
    )
    detailed_acc = detailed_correct / n_eval

    # --- Major field metrics (rolled up from detailed predictions) ---
    major_correct = sum(
        1 for p in has_truth
        if p["predicted_major_field"] == p["true_major_field"]
    )
    major_acc = major_correct / n_eval

    # --- Broad field metrics ---
    broad_correct = sum(
        1 for p in has_truth
        if p["predicted_broad_field"] == p["true_broad_field"]
    )
    broad_acc = broad_correct / n_eval

    # --- Top-k accuracy (detailed) ---
    top_k_detailed_acc = {}
    for k in (3, 5, 10):
        hits = sum(
            1 for p in has_truth
            if p["true_detailed_field"] in p["top_k_detailed_fields"][:k]
        )
        top_k_detailed_acc[k] = hits / n_eval

    # --- Top-k accuracy (major, rolled up) ---
    top_k_major_acc = {}
    for k in (3, 5, 10):
        hits = sum(
            1 for p in has_truth
            if p["true_major_field"] in p["top_k_major_fields"][:k]
        )
        top_k_major_acc[k] = hits / n_eval

    # --- Per-field F1 (detailed) ---
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    for p in has_truth:
        pred = p["predicted_detailed_field"]
        true = p["true_detailed_field"]
        if pred == true:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

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
                "major_field": detailed_to_major.get(fld, ""),
            }
            f1_values.append(f1)

    macro_f1_detailed = float(np.mean(f1_values)) if f1_values else 0.0

    # --- Per-field F1 (major, rolled up) ---
    tp_m: Counter = Counter()
    fp_m: Counter = Counter()
    fn_m: Counter = Counter()
    for p in has_truth:
        pred = p["predicted_major_field"]
        true = p["true_major_field"]
        if pred == true:
            tp_m[true] += 1
        else:
            fp_m[pred] += 1
            fn_m[true] += 1

    major_f1_values = []
    for fld in set(tp_m.keys()) | set(fp_m.keys()) | set(fn_m.keys()):
        prec = tp_m[fld] / (tp_m[fld] + fp_m[fld]) if (tp_m[fld] + fp_m[fld]) > 0 else 0
        rec = tp_m[fld] / (tp_m[fld] + fn_m[fld]) if (tp_m[fld] + fn_m[fld]) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if (tp_m[fld] + fn_m[fld]) > 0:
            major_f1_values.append(f1)
    macro_f1_major = float(np.mean(major_f1_values)) if major_f1_values else 0.0

    # Print results
    print(f"\n  Detailed field accuracy:  {detailed_acc:.4f} ({detailed_correct}/{n_eval})")
    print(f"  Detailed macro F1:       {macro_f1_detailed:.4f}")
    print(f"  Top-3 detailed acc:      {top_k_detailed_acc[3]:.4f}")
    print(f"  Top-5 detailed acc:      {top_k_detailed_acc[5]:.4f}")
    print(f"  Top-10 detailed acc:     {top_k_detailed_acc[10]:.4f}")
    print()
    print(f"  Major field accuracy (rolled up): {major_acc:.4f} ({major_correct}/{n_eval})")
    print(f"  Major macro F1 (rolled up):       {macro_f1_major:.4f}")
    print(f"  Top-3 major acc (rolled up):      {top_k_major_acc[3]:.4f}")
    print(f"  Top-5 major acc (rolled up):      {top_k_major_acc[5]:.4f}")
    print()
    print(f"  Broad field accuracy (rolled up): {broad_acc:.4f} ({broad_correct}/{n_eval})")

    # Worst detailed fields
    worst = sorted(per_field.items(), key=lambda x: x[1]["f1"])[:15]
    print(f"\n  Worst 15 detailed fields by F1:")
    for fld, m in worst:
        print(f"    F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"n={m['support']:3d}  {fld}")

    metrics = {
        "model_name": "detailed_scibert",
        "n_eval": n_eval,
        "detailed_field_accuracy": round(detailed_acc, 4),
        "detailed_macro_f1": round(macro_f1_detailed, 4),
        "major_field_accuracy_rolled_up": round(major_acc, 4),
        "major_macro_f1_rolled_up": round(macro_f1_major, 4),
        "broad_field_accuracy_rolled_up": round(broad_acc, 4),
        "top_k_detailed_accuracy": {str(k): round(v, 4) for k, v in top_k_detailed_acc.items()},
        "top_k_major_accuracy_rolled_up": {str(k): round(v, 4) for k, v in top_k_major_acc.items()},
        "per_detailed_field_metrics": per_field,
    }

    return metrics


def predict_only(
    project_root: Path,
    model_dir: Path,
    test_path: Path,
    output_dir: Path,
    dataset_name: str = "synthetic_test",
    batch_size: int = 32,
    max_length: int = 512,
):
    """Load a saved detailed-field model and run prediction + evaluation."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    taxonomy_path = project_root / "data" / "processed" / "SEDCIP24.json"
    detailed_to_major, detailed_to_broad = build_taxonomy_mappings(taxonomy_path)

    # Load label classes
    with open(model_dir / "label_classes.json") as f:
        label_classes = json.load(f)
    print(f"Loaded model from {model_dir} ({len(label_classes)} classes)")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    if torch.cuda.is_available():
        model = model.cuda()

    predict_and_evaluate(
        project_root=project_root,
        test_path=test_path,
        trainer_or_model=model,
        tokenizer=tokenizer,
        label_classes=label_classes,
        detailed_to_major=detailed_to_major,
        detailed_to_broad=detailed_to_broad,
        output_dir=output_dir,
        dataset_name=dataset_name,
        batch_size=batch_size,
        max_length=max_length,
        model_label=f"detailed_{model_dir.name}",
        use_trainer=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate SciBERT on detailed CIP fields")
    parser.add_argument("--predict-only", action="store_true",
                        help="Load saved model and predict (skip training)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Model directory for predict-only mode")
    parser.add_argument("--train-data", type=str, default="data/generated/train.jsonl",
                        help="Training data JSONL")
    parser.add_argument("--test-data", type=str, default="data/generated/test.jsonl",
                        help="Test data JSONL")
    parser.add_argument("--output-dir", type=str, default="output/predictions",
                        help="Output directory for predictions and metrics")
    parser.add_argument("--model-name", type=str, default="allenai/scibert_scivocab_uncased",
                        help="HF model identifier")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    project_root = find_project_root()
    output_dir = project_root / args.output_dir
    test_path = project_root / args.test_data

    if args.predict_only:
        model_dir = Path(args.model_dir) if args.model_dir else (
            project_root / "output" / "models" / "detailed_finetune"
        )
        predict_only(project_root, model_dir, test_path, output_dir)
    else:
        train_path = project_root / args.train_data

        trainer, tokenizer, label_classes, d2m, d2b = train_detailed_model(
            project_root=project_root,
            train_path=train_path,
            model_name=args.model_name,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        predict_and_evaluate(
            project_root=project_root,
            test_path=test_path,
            trainer_or_model=trainer,
            tokenizer=tokenizer,
            label_classes=label_classes,
            detailed_to_major=d2m,
            detailed_to_broad=d2b,
            output_dir=output_dir,
            dataset_name="synthetic_test",
            model_label=f"detailed_scibert_{args.epochs}ep",
            use_trainer=True,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
