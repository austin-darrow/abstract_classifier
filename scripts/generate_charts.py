"""Generate publication-quality charts for the CIP classifier.

Charts produced:
  1. Learning curve: loss and accuracy per epoch (best model)
  2. Confusion matrix (top-20 fields, synthetic test)
  3. Per-field F1 bar chart (synthetic test)
  4. Confidence distribution histogram
  5. Sweep leaderboard (from sweep results)
  6. LR × epochs heatmap (from sweep results)

Usage:
    # Full charts (GPU required for learning curve — retrains best config):
    python scripts/generate_charts.py

    # Charts from existing predictions only (no GPU needed):
    python scripts/generate_charts.py --no-retrain

    # Specify prediction files:
    python scripts/generate_charts.py --no-retrain \
        --predictions output/predictions/predictions_finetune_scibert_scivocab_uncased_8ep_synthetic_test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_project_root() -> Path:
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return cwd


# =========================================================================
# Chart 1: Learning curve (train loss, val loss, val accuracy per epoch)
# =========================================================================
def generate_learning_curve(project_root: Path, output_dir: Path):
    """Retrain best config with per-epoch logging, plot learning curve."""
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

    print("Training best config for learning curve...")

    # Load data
    train_path = project_root / "data" / "generated" / "train_with_silver.jsonl"
    train_records = []
    with open(train_path) as f:
        for line in f:
            train_records.append(json.loads(line))

    train_texts = [r["abstract"] for r in train_records]
    train_labels = [r["major_field"] for r in train_records]

    le = LabelEncoder()
    le.fit(train_labels)
    y_train = le.transform(train_labels).tolist()
    n_classes = len(le.classes_)

    idx_train, idx_val = train_test_split(
        list(range(len(train_texts))), test_size=0.1, random_state=42, stratify=y_train,
    )
    tr_texts = [train_texts[i] for i in idx_train]
    tr_labels = [y_train[i] for i in idx_train]
    val_texts = [train_texts[i] for i in idx_val]
    val_labels = [y_train[i] for i in idx_val]

    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    train_ds = Dataset.from_dict({"text": tr_texts, "label": tr_labels}).map(tokenize_fn, batched=True)
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(tokenize_fn, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    ckpt_dir = project_root / "output" / "models" / "learning_curve"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=8,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        seed=42,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    log_history = trainer.state.log_history

    # Extract per-epoch metrics
    train_losses = []
    val_losses = []
    val_accs = []
    epochs = []

    for entry in log_history:
        if "eval_loss" in entry:
            val_losses.append(entry["eval_loss"])
            val_accs.append(entry["eval_accuracy"])
            epochs.append(entry["epoch"])
        if "train_loss" in entry and "epoch" in entry:
            train_losses.append(entry["train_loss"])

    # Align lengths
    n = min(len(epochs), len(train_losses), len(val_losses))
    epochs = epochs[:n]
    train_losses = train_losses[:n]
    val_losses = val_losses[:n]
    val_accs = val_accs[:n]

    # Save raw data
    curve_data = {
        "epochs": epochs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_accuracy": val_accs,
    }
    with open(output_dir / "learning_curve_data.json", "w") as f:
        json.dump(curve_data, f, indent=2)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, "o-", label="Train Loss", color="#2196F3")
    ax1.plot(epochs, val_losses, "s-", label="Val Loss", color="#F44336")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_accs, "D-", label="Val Accuracy", color="#4CAF50")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.set_ylim(0.85, 1.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("SciBERT + Silver Labels (lr=3e-5, bs=16)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curve.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'learning_curve.png'}")

    del model, trainer
    torch.cuda.empty_cache()


# =========================================================================
# Chart 2: Confusion matrix (top-N most confused fields)
# =========================================================================
def generate_confusion_matrix(predictions: list[dict], output_dir: Path, top_n: int = 20):
    """Plot confusion matrix for the top-N most common fields."""
    from collections import Counter

    from sklearn.metrics import confusion_matrix

    true_labels = [p["true_major_field"] for p in predictions if p.get("true_major_field")]
    pred_labels = [p["predicted_major_field"] for p in predictions if p.get("true_major_field")]

    # Find top-N fields by support
    field_counts = Counter(true_labels)
    top_fields = [f for f, _ in field_counts.most_common(top_n)]

    # Filter to only those fields
    mask = [t in top_fields for t in true_labels]
    true_filtered = [t for t, m in zip(true_labels, mask) if m]
    pred_filtered = [p for p, m in zip(pred_labels, mask) if m]

    cm = confusion_matrix(true_filtered, pred_filtered, labels=top_fields)
    # Normalize by row
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    # Shorten labels
    short_labels = [f[:35] for f in top_fields]

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(short_labels)))
    ax.set_yticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Normalized Confusion Matrix (Top {top_n} Fields)")
    plt.colorbar(im, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'confusion_matrix.png'}")


# =========================================================================
# Chart 3: Per-field F1 bar chart
# =========================================================================
def generate_per_field_f1(predictions: list[dict], output_dir: Path):
    """Horizontal bar chart of F1 per major field."""
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import LabelEncoder

    true_labels = [p["true_major_field"] for p in predictions if p.get("true_major_field")]
    pred_labels = [p["predicted_major_field"] for p in predictions if p.get("true_major_field")]

    le = LabelEncoder()
    le.fit(true_labels + pred_labels)
    y_true = le.transform(true_labels)
    y_pred = le.transform(pred_labels)

    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=range(len(le.classes_)))
    fields = le.classes_

    # Sort by F1
    sorted_idx = np.argsort(per_class_f1)
    sorted_fields = [fields[i][:40] for i in sorted_idx]
    sorted_f1 = [per_class_f1[i] for i in sorted_idx]

    # Color by F1 value
    colors = ["#F44336" if f < 0.5 else "#FF9800" if f < 0.8 else "#4CAF50" for f in sorted_f1]

    fig, ax = plt.subplots(figsize=(12, max(8, len(fields) * 0.25)))
    ax.barh(range(len(sorted_fields)), sorted_f1, color=colors)
    ax.set_yticks(range(len(sorted_fields)))
    ax.set_yticklabels(sorted_fields, fontsize=7)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Field F1 Score (Synthetic Test)")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.8, color="gray", linestyle="--", alpha=0.5, label="0.8 threshold")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    plt.savefig(output_dir / "per_field_f1.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'per_field_f1.png'}")


# =========================================================================
# Chart 4: Confidence distribution
# =========================================================================
def generate_confidence_dist(predictions: list[dict], output_dir: Path):
    """Histogram of prediction confidence, colored by correct/incorrect."""
    true_labels = [p["true_major_field"] for p in predictions if p.get("true_major_field")]
    pred_labels = [p["predicted_major_field"] for p in predictions if p.get("true_major_field")]
    confidences = [p["confidence"] for p in predictions if p.get("true_major_field")]

    correct_conf = [c for t, p, c in zip(true_labels, pred_labels, confidences) if t == p]
    wrong_conf = [c for t, p, c in zip(true_labels, pred_labels, confidences) if t != p]

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 50)
    ax.hist(correct_conf, bins=bins, alpha=0.7, label=f"Correct (n={len(correct_conf)})", color="#4CAF50")
    ax.hist(wrong_conf, bins=bins, alpha=0.7, label=f"Incorrect (n={len(wrong_conf)})", color="#F44336")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'confidence_distribution.png'}")


# =========================================================================
# Chart 5: Sweep leaderboard (from sweep results)
# =========================================================================
def generate_sweep_charts(project_root: Path, output_dir: Path):
    """Generate charts from sweep results if available."""
    sweep_path = project_root / "output" / "sweep" / "sweep_results.jsonl"
    if not sweep_path.exists():
        print("  No sweep results found, skipping sweep charts")
        return

    results = []
    with open(sweep_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "completed":
                results.append(r)

    results.sort(key=lambda r: r["synthetic_test"]["major_accuracy"], reverse=True)

    # Leaderboard bar chart (top 15)
    top15 = results[:15]
    names = [r["config"]["name"][:40] for r in top15]
    majors = [r["synthetic_test"]["major_accuracy"] for r in top15]
    f1s = [r["synthetic_test"]["macro_f1"] for r in top15]

    fig, ax = plt.subplots(figsize=(12, 7))
    y = range(len(names))
    width = 0.35
    ax.barh([i - width / 2 for i in y], majors, width, label="Major Accuracy", color="#2196F3")
    ax.barh([i + width / 2 for i in y], f1s, width, label="Macro F1", color="#FF9800")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Score")
    ax.set_title("C4 Sweep: Top 15 Configurations")
    ax.legend()
    ax.set_xlim(0.85, 0.96)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_leaderboard.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'sweep_leaderboard.png'}")

    # LR × epochs heatmap (SciBERT, default regularization only)
    scibert = [r for r in results if "scibert" in r["config"]["model_name"]
               and r["config"]["label_smoothing"] == 0.0
               and r["config"]["scheduler_type"] == "linear"
               and r["config"]["gradient_accumulation_steps"] == 1
               and r["config"]["freeze_layers"] == 0
               and r["config"]["weight_decay"] == 0.01]
    if scibert:
        lrs = sorted(set(r["config"]["lr"] for r in scibert))
        epoch_vals = sorted(set(r["config"]["epochs"] for r in scibert))

        heatmap = np.zeros((len(lrs), len(epoch_vals)))
        for r in scibert:
            li = lrs.index(r["config"]["lr"])
            ei = epoch_vals.index(r["config"]["epochs"])
            heatmap[li][ei] = r["synthetic_test"]["major_accuracy"]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto",
                        vmin=max(0.88, heatmap[heatmap > 0].min() - 0.01),
                        vmax=heatmap.max() + 0.005)
        ax.set_xticks(range(len(epoch_vals)))
        ax.set_xticklabels(epoch_vals)
        ax.set_yticks(range(len(lrs)))
        ax.set_yticklabels([f"{lr:.0e}" for lr in lrs])
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Learning Rate")
        ax.set_title("SciBERT Major Accuracy: LR × Epochs")
        for i in range(len(lrs)):
            for j in range(len(epoch_vals)):
                if heatmap[i][j] > 0:
                    ax.text(j, i, f"{heatmap[i][j]:.3f}", ha="center", va="center", fontsize=10)
        plt.colorbar(im, shrink=0.8)
        plt.tight_layout()
        plt.savefig(output_dir / "lr_epochs_heatmap.png", dpi=150)
        plt.close()
        print(f"  Saved: {output_dir / 'lr_epochs_heatmap.png'}")


# =========================================================================
# Chart 6: Approach progression (B0 → C4)
# =========================================================================
def generate_progression_chart(output_dir: Path):
    """Bar chart showing accuracy improvement across approaches."""
    approaches = [
        ("B0\nFAISS", 0.5224, 0.4363),
        ("B1\nkNN", 0.8564, 0.8500),
        ("B2\nTF-IDF", 0.8219, 0.8403),
        ("B3\nEmbed", 0.8858, 0.8701),
        ("B4\nSetFit", 0.9188, 0.8448),
        ("B5\nSciBERT", 0.9077, 0.8428),
        ("C3\n+Silver", 0.9280, 0.9049),
        ("C4\nTuned", 0.9396, 0.9369),
    ]

    names = [a[0] for a in approaches]
    accs = [a[1] for a in approaches]
    f1s = [a[2] for a in approaches]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width / 2, accs, width, label="Major Accuracy", color="#2196F3")
    bars2 = ax.bar(x + width / 2, f1s, width, label="Macro F1", color="#FF9800")

    ax.set_ylabel("Score")
    ax.set_title("CIP Classifier: Approach Progression (Synthetic Test)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend()
    ax.set_ylim(0.3, 1.0)
    ax.grid(True, alpha=0.2, axis="y")

    # Annotate best
    ax.annotate("93.96%", xy=(7 - width / 2, 0.9396), ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "approach_progression.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'approach_progression.png'}")


def main():
    parser = argparse.ArgumentParser(description="Generate charts for CIP classifier results")
    parser.add_argument("--no-retrain", action="store_true",
                        help="Skip learning curve (no GPU needed)")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to prediction JSON for confusion/F1/confidence charts")
    parser.add_argument("--output-dir", type=str, default="output/charts",
                        help="Output directory for charts")
    args = parser.parse_args()

    project_root = find_project_root()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating charts in: {output_dir}\n")

    # Find prediction file
    pred_path = None
    if args.predictions:
        pred_path = project_root / args.predictions
    else:
        # Auto-find best synthetic test predictions
        candidates = [
            "output/predictions/predictions_finetune_scibert_scivocab_uncased_8ep_synthetic_test.json",
            "output/predictions/predictions_finetune_scibert_scivocab_uncased_5ep_synthetic_test.json",
        ]
        for c in candidates:
            p = project_root / c
            if p.exists():
                pred_path = p
                break

    # Chart 1: Learning curve (requires GPU)
    if not args.no_retrain:
        print("=" * 50)
        print("Chart 1: Learning Curve")
        print("=" * 50)
        generate_learning_curve(project_root, output_dir)
    else:
        # Check if learning curve data already exists
        lc_data = output_dir / "learning_curve_data.json"
        if lc_data.exists():
            print("Learning curve data found, plotting from saved data...")
            with open(lc_data) as f:
                curve = json.load(f)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.plot(curve["epochs"], curve["train_loss"], "o-", label="Train Loss", color="#2196F3")
            ax1.plot(curve["epochs"], curve["val_loss"], "s-", label="Val Loss", color="#F44336")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training & Validation Loss")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(curve["epochs"], curve["val_accuracy"], "D-", label="Val Accuracy", color="#4CAF50")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.set_title("Validation Accuracy")
            ax2.set_ylim(0.85, 1.0)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            fig.suptitle("SciBERT + Silver Labels (lr=3e-5, bs=16)", fontsize=13)
            plt.tight_layout()
            plt.savefig(output_dir / "learning_curve.png", dpi=150)
            plt.close()
            print(f"  Saved: {output_dir / 'learning_curve.png'}")
        else:
            print("Skipping learning curve (use without --no-retrain on GPU node)")

    # Charts 2-4: From predictions
    if pred_path and pred_path.exists():
        with open(pred_path) as f:
            data = json.load(f)
        predictions = data.get("predictions", data)
        print(f"\nUsing predictions from: {pred_path.name} ({len(predictions)} records)")

        print("\n" + "=" * 50)
        print("Chart 2: Confusion Matrix")
        print("=" * 50)
        generate_confusion_matrix(predictions, output_dir)

        print("\n" + "=" * 50)
        print("Chart 3: Per-Field F1")
        print("=" * 50)
        generate_per_field_f1(predictions, output_dir)

        print("\n" + "=" * 50)
        print("Chart 4: Confidence Distribution")
        print("=" * 50)
        generate_confidence_dist(predictions, output_dir)
    else:
        print("\nNo prediction file found, skipping confusion/F1/confidence charts")
        print("  Run with: --predictions output/predictions/predictions_*.json")

    # Chart 5: Sweep results
    print("\n" + "=" * 50)
    print("Chart 5: Sweep Leaderboard + Heatmap")
    print("=" * 50)
    generate_sweep_charts(project_root, output_dir)

    # Chart 6: Approach progression
    print("\n" + "=" * 50)
    print("Chart 6: Approach Progression")
    print("=" * 50)
    generate_progression_chart(output_dir)

    print(f"\n{'='*50}")
    print(f"All charts saved to: {output_dir}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
