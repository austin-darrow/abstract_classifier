"""B5: Full encoder fine-tune (SciBERT / DeBERTa) with classification head.

Uses HuggingFace Trainer for standard text classification fine-tuning.
Slowest to train but typically strongest single-model approach.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import PipelineConfig
from ..utils import load_json


def finetune_classify(
    cfg: PipelineConfig,
    project_root: Path,
    train_path: Path | None = None,
    test_path: Path | None = None,
    dataset_name: str = "synthetic_test",
    model_name_or_path: str | None = None,
    lr: float = 2e-5,
    epochs: int = 3,
    batch_size: int = 16,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_length: int = 512,
    early_stopping_patience: int = 3,
    freeze_layers: int = 0,
    seed: int = 42,
) -> "PredictionSet":
    """Fine-tune a transformer encoder for classification.

    Args:
        cfg: Pipeline config.
        project_root: Project root directory.
        train_path: Path to training JSONL.
        test_path: Path to test JSONL or Excel.
        dataset_name: Name for the prediction set.
        model_name_or_path: HF model identifier (default: allenai/scibert_scivocab_uncased).
        lr: Learning rate.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        warmup_ratio: Fraction of steps for warmup.
        weight_decay: Weight decay.
        max_length: Max token length.
        early_stopping_patience: Early stopping patience.

    Returns:
        PredictionSet with predictions for all test abstracts.
    """
    import torch
    from datasets import Dataset
    from sklearn.preprocessing import LabelEncoder
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    from ..evaluation.predictions import Prediction, PredictionSet
    from .faiss_retrieval import _load_jsonl, _load_test_data

    if model_name_or_path is None:
        model_name_or_path = "allenai/scibert_scivocab_uncased"

    # Build major→broad mapping
    taxonomy_path = cfg.resolve_path(cfg.paths.taxonomy_json, project_root)
    taxonomy = load_json(taxonomy_path)
    major_to_broad = {}
    for entry in taxonomy:
        major_to_broad.setdefault(entry["Major_Field_label"], entry["Broad_Field_label"])

    # Load data
    if train_path is None:
        train_path = cfg.resolve_path(cfg.train.train_data, project_root)
    train_records = _load_jsonl(train_path)
    print(f"Loaded {len(train_records)} training abstracts")

    if test_path is None:
        test_path = cfg.resolve_path(cfg.train.test_data, project_root)
    test_records = _load_test_data(test_path, major_to_broad)
    print(f"Loaded {len(test_records)} test abstracts")

    train_texts = [r["abstract"] for r in train_records]
    train_labels = [r["major_field"] for r in train_records]
    test_texts = [r["abstract"] for r in test_records]

    # Encode labels
    le = LabelEncoder()
    le.fit(train_labels)
    y_train = le.transform(train_labels).tolist()
    n_classes = len(le.classes_)

    # Create 90/10 train/val split for early stopping
    from sklearn.model_selection import train_test_split
    idx_train, idx_val = train_test_split(
        list(range(len(train_texts))), test_size=0.1, random_state=42, stratify=y_train,
    )
    val_texts = [train_texts[i] for i in idx_val]
    val_labels = [y_train[i] for i in idx_val]
    tr_texts = [train_texts[i] for i in idx_train]
    tr_labels = [y_train[i] for i in idx_train]

    # Tokenize
    print(f"Loading tokenizer + model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=n_classes,
    )

    # Freeze early encoder layers if requested
    if freeze_layers > 0 and hasattr(model, "bert"):
        embeddings = model.bert.embeddings
        for param in embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(model.bert.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"  Froze embeddings + first {freeze_layers} encoder layers")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")

    train_ds = Dataset.from_dict({"text": tr_texts, "label": tr_labels}).map(tokenize_fn, batched=True)
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(tokenize_fn, batched=True)
    test_ds = Dataset.from_dict({"text": test_texts, "label": [0] * len(test_texts)}).map(tokenize_fn, batched=True)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask"])

    # Training
    output_dir = project_root / "output" / "models" / "finetune"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
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
    )

    def compute_hf_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_hf_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    print(f"Training {model_name_or_path} (epochs={epochs}, lr={lr}, bs={batch_size})...")
    trainer.train()
    print("Training complete.")

    # Predict on test set
    print(f"Predicting on {len(test_texts)} test abstracts...")
    raw_preds = trainer.predict(test_ds)
    logits = raw_preds.predictions
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    y_pred = probs.argmax(axis=1)
    pred_labels_arr = le.inverse_transform(y_pred)

    # Build predictions
    predictions = []
    for i in range(len(test_records)):
        predicted_major = pred_labels_arr[i]
        predicted_broad = major_to_broad.get(predicted_major, "")
        confidence = float(probs[i].max())

        top_k_idx = np.argsort(probs[i])[::-1][:10]
        top_k_fields = [le.classes_[idx] for idx in top_k_idx]
        top_k_scores = [float(probs[i][idx]) for idx in top_k_idx]

        predictions.append(Prediction(
            abstract=test_records[i]["abstract"],
            true_major_field=test_records[i].get("major_field", ""),
            true_broad_field=test_records[i].get("broad_field", ""),
            predicted_major_field=predicted_major,
            predicted_broad_field=predicted_broad,
            confidence=confidence,
            top_k_major_fields=top_k_fields,
            top_k_scores=top_k_scores,
        ))

    model_label = model_name_or_path.split("/")[-1]
    pred_set = PredictionSet(
        model_name=f"finetune_{model_label}_{epochs}ep",
        predictions=predictions,
        dataset=dataset_name,
        metadata={
            "model": model_name_or_path,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "max_length": max_length,
            "n_classes": n_classes,
            "n_train": len(tr_texts),
            "n_val": len(val_texts),
            "n_test": len(test_records),
        },
    )

    print(f"Fine-tune classification complete: {len(predictions)} predictions")
    return pred_set


def finetune_predict(
    cfg: PipelineConfig,
    project_root: Path,
    model_path: Path,
    test_path: Path | None = None,
    dataset_name: str = "synthetic_test",
    max_length: int = 512,
    batch_size: int = 32,
) -> "PredictionSet":
    """Load a saved fine-tuned model and predict on a test set.

    Args:
        cfg: Pipeline config.
        project_root: Project root directory.
        model_path: Path to saved model directory (contains model + tokenizer + label_classes.json).
        test_path: Path to test JSONL or Excel.
        dataset_name: Name for the prediction set.
        max_length: Max token length.
        batch_size: Inference batch size.

    Returns:
        PredictionSet with predictions.
    """
    import json as _json

    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from ..evaluation.predictions import Prediction, PredictionSet
    from .faiss_retrieval import _load_test_data

    # Build major→broad mapping
    taxonomy_path = cfg.resolve_path(cfg.paths.taxonomy_json, project_root)
    taxonomy = load_json(taxonomy_path)
    major_to_broad = {}
    for entry in taxonomy:
        major_to_broad.setdefault(entry["Major_Field_label"], entry["Broad_Field_label"])

    # Load label encoder
    label_classes_path = model_path / "label_classes.json"
    with open(label_classes_path) as f:
        classes = _json.load(f)
    print(f"Loaded model from {model_path} ({len(classes)} classes)")

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # Load test data
    if test_path is None:
        test_path = cfg.resolve_path(cfg.train.test_data, project_root)
    test_records = _load_test_data(test_path, major_to_broad)
    print(f"Loaded {len(test_records)} test abstracts")
    test_texts = [r["abstract"] for r in test_records]

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")

    test_ds = Dataset.from_dict({"text": test_texts, "label": [0] * len(test_texts)}).map(tokenize_fn, batched=True)
    test_ds.set_format("torch", columns=["input_ids", "attention_mask"])

    # Predict in batches
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(test_ds), batch_size):
            batch = test_ds[i:i + batch_size]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)
    y_pred = probs.argmax(axis=1)

    # Build predictions
    predictions = []
    for i in range(len(test_records)):
        predicted_major = classes[y_pred[i]]
        predicted_broad = major_to_broad.get(predicted_major, "")
        confidence = float(probs[i].max())

        top_k_idx = np.argsort(probs[i])[::-1][:10]
        top_k_fields = [classes[idx] for idx in top_k_idx]
        top_k_scores = [float(probs[i][idx]) for idx in top_k_idx]

        predictions.append(Prediction(
            abstract=test_records[i]["abstract"],
            true_major_field=test_records[i].get("major_field", ""),
            true_broad_field=test_records[i].get("broad_field", ""),
            predicted_major_field=predicted_major,
            predicted_broad_field=predicted_broad,
            confidence=confidence,
            top_k_major_fields=top_k_fields,
            top_k_scores=top_k_scores,
        ))

    model_label = model_path.name
    pred_set = PredictionSet(
        model_name=f"finetune_{model_label}",
        predictions=predictions,
        dataset=dataset_name,
        metadata={
            "model_path": str(model_path),
            "n_classes": len(classes),
            "n_test": len(test_records),
        },
    )

    print(f"Prediction complete: {len(predictions)} predictions")
    return pred_set
