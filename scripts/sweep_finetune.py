"""C4: Exhaustive hyperparameter sweep for SciBERT fine-tuning.

Explores models, learning rates, epochs, regularization, loss functions,
training strategies, and self-training rounds to maximize performance.

Usage:
    # Full sweep (GPU required):
    python scripts/sweep_finetune.py --train-data data/generated/train_with_silver.jsonl

    # Quick test (2 configs):
    python scripts/sweep_finetune.py --train-data data/generated/train_with_silver.jsonl --quick

    # Single config by index:
    python scripts/sweep_finetune.py --train-data data/generated/train_with_silver.jsonl --config-idx 5

    # Resume from a checkpoint (skip completed):
    python scripts/sweep_finetune.py --train-data data/generated/train_with_silver.jsonl --resume
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path

import numpy as np


@dataclass
class SweepConfig:
    """A single hyperparameter configuration to evaluate."""

    model_name: str = "allenai/scibert_scivocab_uncased"
    lr: float = 2e-5
    epochs: int = 5
    batch_size: int = 16
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_length: int = 512
    label_smoothing: float = 0.0
    scheduler_type: str = "linear"  # linear, cosine, cosine_with_restarts
    gradient_accumulation_steps: int = 1
    freeze_layers: int = 0  # freeze first N encoder layers
    dropout: float = 0.1
    early_stopping_patience: int = 3
    # Data strategy
    train_data: str = "data/generated/train_with_silver.jsonl"
    # Identifiers
    name: str = ""

    def __post_init__(self):
        if not self.name:
            model_short = self.model_name.split("/")[-1]
            self.name = (
                f"{model_short}_lr{self.lr:.0e}_ep{self.epochs}"
                f"_bs{self.batch_size * self.gradient_accumulation_steps}"
                f"_ls{self.label_smoothing}_wd{self.weight_decay}"
                f"_{self.scheduler_type}"
            )
            if self.freeze_layers > 0:
                self.name += f"_freeze{self.freeze_layers}"


def build_sweep_configs() -> list[SweepConfig]:
    """Build the full set of configurations to sweep."""
    configs = []

    # =========================================================================
    # TIER 1: Core hyperparameters (most impactful)
    # =========================================================================

    # Learning rate × epochs grid (the two most impactful hyperparams)
    for lr, epochs in product(
        [1e-5, 2e-5, 3e-5, 5e-5],
        [3, 5, 8, 10],
    ):
        configs.append(SweepConfig(lr=lr, epochs=epochs))

    # =========================================================================
    # TIER 2: Regularization
    # =========================================================================

    # Label smoothing (reduces overconfidence, helps with noisy labels)
    for ls in [0.05, 0.1, 0.15, 0.2]:
        configs.append(SweepConfig(lr=2e-5, epochs=5, label_smoothing=ls))

    # Weight decay
    for wd in [0.005, 0.05, 0.1]:
        configs.append(SweepConfig(lr=2e-5, epochs=5, weight_decay=wd))

    # =========================================================================
    # TIER 3: Scheduler
    # =========================================================================

    for sched in ["cosine", "cosine_with_restarts"]:
        configs.append(SweepConfig(lr=2e-5, epochs=5, scheduler_type=sched))
        configs.append(SweepConfig(lr=3e-5, epochs=8, scheduler_type=sched))

    # =========================================================================
    # TIER 4: Effective batch size (gradient accumulation)
    # =========================================================================

    # Larger effective batch (32, 64) via accumulation
    for accum in [2, 4]:
        configs.append(SweepConfig(lr=2e-5, epochs=5, gradient_accumulation_steps=accum))
        configs.append(SweepConfig(lr=3e-5, epochs=5, gradient_accumulation_steps=accum))

    # =========================================================================
    # TIER 5: Layer freezing (transfer learning strategy)
    # =========================================================================

    # Freeze early layers — keep general language understanding, tune top layers
    for freeze in [4, 6, 8]:
        configs.append(SweepConfig(lr=3e-5, epochs=8, freeze_layers=freeze))

    # =========================================================================
    # TIER 6: Alternative models
    # =========================================================================

    # DeBERTa-v3-base: better architecture (disentangled attention + enhanced mask decoder)
    for lr, epochs in [(1e-5, 5), (2e-5, 5), (2e-5, 8), (3e-5, 5)]:
        configs.append(SweepConfig(
            model_name="microsoft/deberta-v3-base", lr=lr, epochs=epochs,
        ))

    # DeBERTa-v3-large: 304M params, potentially much stronger
    for lr, epochs in [(5e-6, 5), (1e-5, 5), (1e-5, 8)]:
        configs.append(SweepConfig(
            model_name="microsoft/deberta-v3-large", lr=lr, epochs=epochs,
            batch_size=8, gradient_accumulation_steps=2,
        ))

    # SPECTER2: removed — adapter-based model, not compatible with AutoModelForSequenceClassification

    # BiomedBERT: biomedical-domain pre-training (strong on life-science abstracts)
    for lr, epochs in [(2e-5, 5), (3e-5, 5), (2e-5, 8)]:
        configs.append(SweepConfig(
            model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", lr=lr, epochs=epochs,
        ))

    # =========================================================================
    # TIER 7: Best-of combinations (fill in after Tier 1 results)
    # =========================================================================

    # Pre-defined "likely best" combos based on literature
    configs.append(SweepConfig(
        lr=2e-5, epochs=8, label_smoothing=0.1,
        scheduler_type="cosine", weight_decay=0.01,
        name="scibert_best_guess_v1",
    ))
    configs.append(SweepConfig(
        lr=3e-5, epochs=5, label_smoothing=0.05,
        scheduler_type="cosine", weight_decay=0.05,
        name="scibert_best_guess_v2",
    ))
    configs.append(SweepConfig(
        model_name="microsoft/deberta-v3-base",
        lr=2e-5, epochs=8, label_smoothing=0.1,
        scheduler_type="cosine",
        name="deberta_best_guess",
    ))

    # Deduplicate by name
    seen = set()
    deduped = []
    for c in configs:
        if c.name not in seen:
            seen.add(c.name)
            deduped.append(c)
    return deduped


def train_single_config(
    cfg: SweepConfig,
    project_root: Path,
    output_dir: Path,
) -> dict:
    """Train and evaluate a single configuration. Returns results dict."""
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

    results = {"config": asdict(cfg), "status": "started", "timestamp": time.time()}
    start_time = time.time()

    try:
        # Load taxonomy
        taxonomy_path = project_root / "data" / "processed" / "SEDCIP24.json"
        with open(taxonomy_path) as f:
            taxonomy = json.load(f)
        major_to_broad = {}
        for entry in taxonomy:
            major_to_broad.setdefault(entry["Major_Field_label"], entry["Broad_Field_label"])

        # Load training data
        train_path = project_root / cfg.train_data
        train_records = []
        with open(train_path) as f:
            for line in f:
                train_records.append(json.loads(line))
        print(f"  Loaded {len(train_records)} training records")

        train_texts = [r["abstract"] for r in train_records]
        train_labels = [r["major_field"] for r in train_records]

        # Encode labels
        le = LabelEncoder()
        le.fit(train_labels)
        y_train = le.transform(train_labels).tolist()
        n_classes = len(le.classes_)

        # Train/val split
        idx_train, idx_val = train_test_split(
            list(range(len(train_texts))), test_size=0.1, random_state=42, stratify=y_train,
        )
        tr_texts = [train_texts[i] for i in idx_train]
        tr_labels = [y_train[i] for i in idx_train]
        val_texts = [train_texts[i] for i in idx_val]
        val_labels = [y_train[i] for i in idx_val]

        # Load model + tokenizer
        print(f"  Loading model: {cfg.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=n_classes,
        )

        # Apply layer freezing
        if cfg.freeze_layers > 0:
            encoder = None
            if hasattr(model, "bert"):
                encoder = model.bert.encoder
            elif hasattr(model, "deberta"):
                encoder = model.deberta.encoder
            if encoder and hasattr(encoder, "layer"):
                for i, layer in enumerate(encoder.layer):
                    if i < cfg.freeze_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                print(f"  Froze first {cfg.freeze_layers} encoder layers")

        # Apply custom dropout if different from default
        if cfg.dropout != 0.1:
            if hasattr(model.config, "hidden_dropout_prob"):
                model.config.hidden_dropout_prob = cfg.dropout
            if hasattr(model.config, "attention_probs_dropout_prob"):
                model.config.attention_probs_dropout_prob = cfg.dropout

        # Tokenize
        def tokenize_fn(examples):
            return tokenizer(
                examples["text"], truncation=True,
                max_length=cfg.max_length, padding="max_length",
            )

        train_ds = Dataset.from_dict({"text": tr_texts, "label": tr_labels}).map(tokenize_fn, batched=True)
        val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(tokenize_fn, batched=True)
        train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        # Training args
        model_output_dir = output_dir / "checkpoints" / cfg.name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(model_output_dir),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size * 2,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.lr,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            lr_scheduler_type=cfg.scheduler_type,
            label_smoothing_factor=cfg.label_smoothing,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            logging_steps=100,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
            report_to="none",
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            acc = (preds == labels).mean()
            return {"accuracy": acc}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
        )

        # Train
        print(f"  Training (epochs={cfg.epochs}, lr={cfg.lr}, bs={cfg.batch_size}×{cfg.gradient_accumulation_steps})...")
        train_result = trainer.train()
        train_time = time.time() - start_time

        # Evaluate on validation set
        val_metrics = trainer.evaluate()

        # Evaluate on synthetic test
        test_path = project_root / "data" / "generated" / "test.jsonl"
        test_records = []
        with open(test_path) as f:
            for line in f:
                test_records.append(json.loads(line))
        test_texts = [r["abstract"] for r in test_records]
        test_labels_str = [r["major_field"] for r in test_records]
        test_labels = le.transform(test_labels_str).tolist()

        test_ds = Dataset.from_dict({"text": test_texts, "label": test_labels}).map(tokenize_fn, batched=True)
        test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        test_preds = trainer.predict(test_ds)
        test_logits = test_preds.predictions
        test_probs = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
        test_pred_labels = test_probs.argmax(axis=1)

        # Compute metrics
        test_acc_major = (test_pred_labels == np.array(test_labels)).mean()

        # Broad accuracy
        broad_correct = 0
        for i in range(len(test_records)):
            pred_major = le.classes_[test_pred_labels[i]]
            true_major = test_labels_str[i]
            pred_broad = major_to_broad.get(pred_major, "")
            true_broad = major_to_broad.get(true_major, "")
            if pred_broad == true_broad:
                broad_correct += 1
        test_acc_broad = broad_correct / len(test_records)

        # Top-3 and Top-5
        top3_correct = 0
        top5_correct = 0
        for i in range(len(test_records)):
            top_k_idx = np.argsort(test_probs[i])[::-1]
            top3_fields = [le.classes_[idx] for idx in top_k_idx[:3]]
            top5_fields = [le.classes_[idx] for idx in top_k_idx[:5]]
            if test_labels_str[i] in top3_fields:
                top3_correct += 1
            if test_labels_str[i] in top5_fields:
                top5_correct += 1
        top3_acc = top3_correct / len(test_records)
        top5_acc = top5_correct / len(test_records)

        # Per-class F1 (macro)
        from sklearn.metrics import f1_score, classification_report
        macro_f1 = f1_score(test_labels, test_pred_labels, average="macro")

        # Per-field F1 (find worst fields)
        per_class_f1 = f1_score(test_labels, test_pred_labels, average=None)
        worst_fields = []
        sorted_idx = np.argsort(per_class_f1)
        for idx in sorted_idx[:10]:
            field_name = le.classes_[idx]
            support = sum(1 for l in test_labels if l == idx)
            worst_fields.append({
                "field": field_name,
                "f1": float(per_class_f1[idx]),
                "support": support,
            })

        # Confidence stats
        confidences = test_probs.max(axis=1)
        avg_confidence = float(confidences.mean())

        results.update({
            "status": "completed",
            "train_time_seconds": train_time,
            "val_loss": val_metrics["eval_loss"],
            "val_accuracy": val_metrics["eval_accuracy"],
            "synthetic_test": {
                "major_accuracy": float(test_acc_major),
                "broad_accuracy": float(test_acc_broad),
                "macro_f1": float(macro_f1),
                "top3_accuracy": float(top3_acc),
                "top5_accuracy": float(top5_acc),
                "avg_confidence": avg_confidence,
                "worst_fields": worst_fields,
            },
            "best_epoch": trainer.state.best_metric,
            "total_steps": trainer.state.global_step,
        })

        # Save model + label encoder for top configs (can re-evaluate on real TACC later)
        if test_acc_major >= 0.93:  # Only save if beating current best (0.928)
            save_dir = output_dir / "models" / cfg.name
            save_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            with open(save_dir / "label_classes.json", "w") as f:
                json.dump(le.classes_.tolist(), f)
            print(f"  ★ Model saved (beat 0.93 threshold): {save_dir}")

        print(f"  ✓ {cfg.name}: major={test_acc_major:.4f}, broad={test_acc_broad:.4f}, "
              f"macro_f1={macro_f1:.4f}, time={train_time:.0f}s")

        # Clean up GPU memory
        del model, trainer
        torch.cuda.empty_cache()

    except Exception as e:
        # Retry with halved batch size on OOM
        if "out of memory" in str(e).lower() and cfg.batch_size > 4:
            print(f"  ⚠ OOM detected, retrying with batch_size={cfg.batch_size // 2}...")
            import torch
            torch.cuda.empty_cache()
            cfg.batch_size = cfg.batch_size // 2
            return train_single_config(cfg, project_root, output_dir)

        results.update({
            "status": "failed",
            "error": str(e),
            "train_time_seconds": time.time() - start_time,
        })
        print(f"  ✗ {cfg.name}: FAILED - {e}")
        import torch
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="C4: Hyperparameter sweep for SciBERT fine-tuning")
    parser.add_argument("--train-data", type=str, default="data/generated/train_with_silver.jsonl",
                        help="Training data path relative to project root")
    parser.add_argument("--quick", action="store_true",
                        help="Run only 2 configs for testing")
    parser.add_argument("--config-idx", type=int, default=None,
                        help="Run a single config by index (for SLURM array jobs)")
    parser.add_argument("--tier", type=int, default=None,
                        help="Run only configs from a specific tier (1-7)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs that already have results")
    parser.add_argument("--output-dir", type=str, default="output/sweep",
                        help="Output directory for sweep results")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build configs
    all_configs = build_sweep_configs()

    # Override train data path
    for c in all_configs:
        c.train_data = args.train_data

    # Filter by tier if requested
    if args.tier is not None:
        tier_ranges = {
            1: (0, 16),    # LR × epochs grid (4×4=16)
            2: (16, 23),   # Label smoothing (4) + weight decay (3)
            3: (23, 27),   # Schedulers (4)
            4: (27, 31),   # Gradient accumulation (4)
            5: (31, 34),   # Layer freezing (3)
            6: (34, 48),   # Alternative models (~14)
            7: (48, None), # Best-of combos
        }
        start, end = tier_ranges.get(args.tier, (0, None))
        all_configs = all_configs[start:end]
        print(f"Running tier {args.tier}: {len(all_configs)} configs")

    if args.quick:
        all_configs = all_configs[:2]

    if args.config_idx is not None:
        if args.config_idx >= len(all_configs):
            print(f"Config index {args.config_idx} out of range (max {len(all_configs) - 1})")
            return
        all_configs = [all_configs[args.config_idx]]

    # Load existing results for resume
    results_file = output_dir / "sweep_results.jsonl"
    completed_names = set()
    if args.resume and results_file.exists():
        with open(results_file) as f:
            for line in f:
                r = json.loads(line)
                if r.get("status") == "completed":
                    completed_names.add(r["config"]["name"])
        print(f"Resuming: {len(completed_names)} configs already completed")

    # Print sweep plan
    print(f"\n{'='*70}")
    print(f"C4 HYPERPARAMETER SWEEP")
    print(f"{'='*70}")
    print(f"Total configs: {len(all_configs)}")
    print(f"Output: {output_dir}")
    print(f"Train data: {args.train_data}")
    print(f"\nConfigs:")
    for i, c in enumerate(all_configs):
        skip = "SKIP" if c.name in completed_names else ""
        print(f"  [{i:3d}] {c.name} {skip}")
    print(f"{'='*70}\n")

    # Run sweep
    all_results = []
    for i, cfg in enumerate(all_configs):
        if cfg.name in completed_names:
            print(f"[{i+1}/{len(all_configs)}] Skipping {cfg.name} (already completed)")
            continue

        print(f"\n[{i+1}/{len(all_configs)}] Running: {cfg.name}")
        print(f"  Model: {cfg.model_name}, LR: {cfg.lr}, Epochs: {cfg.epochs}, "
              f"BS: {cfg.batch_size}×{cfg.gradient_accumulation_steps}, "
              f"LS: {cfg.label_smoothing}, Sched: {cfg.scheduler_type}")

        result = train_single_config(cfg, project_root, output_dir)
        all_results.append(result)

        # Append to results file incrementally
        with open(results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    # Print summary
    print(f"\n{'='*70}")
    print("SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<50} {'Major':>7} {'Broad':>7} {'F1':>7} {'Time':>6}")
    print("-" * 80)

    # Load all results (including previously completed)
    if results_file.exists():
        final_results = []
        with open(results_file) as f:
            for line in f:
                final_results.append(json.loads(line))

        # Sort by major accuracy
        completed = [r for r in final_results if r["status"] == "completed"]
        completed.sort(key=lambda r: r["synthetic_test"]["major_accuracy"], reverse=True)

        for r in completed[:20]:  # Top 20
            name = r["config"]["name"][:50]
            major = r["synthetic_test"]["major_accuracy"]
            broad = r["synthetic_test"]["broad_accuracy"]
            f1 = r["synthetic_test"]["macro_f1"]
            t = r["train_time_seconds"]
            print(f"{name:<50} {major:>7.4f} {broad:>7.4f} {f1:>7.4f} {t:>5.0f}s")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
