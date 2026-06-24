"""B4: SetFit — contrastive fine-tuning with few examples per class.

Uses sentence-transformers SetFit library: contrastive pairs → adapter → LogReg head.
Particularly effective in low-data regimes, which makes it interesting for domain shift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import PipelineConfig
from ..utils import load_json


def setfit_train(
    cfg: PipelineConfig,
    project_root: Path,
    train_path: Path | None = None,
    num_iterations: int | None = None,
    num_epochs: int | None = None,
    batch_size: int = 16,
    max_samples_per_class: int | None = None,
    use_bf16: bool = False,
) -> tuple:
    """Train a SetFit model. Returns (model, label_encoder, major_to_broad, metadata).

    Args:
        cfg: Pipeline config.
        project_root: Project root directory.
        train_path: Path to training JSONL.
        num_iterations: Number of contrastive training iterations.
        num_epochs: Number of fine-tuning epochs.
        batch_size: Training batch size for contrastive pairs.
        max_samples_per_class: Cap samples per class (None = all data).
        use_bf16: Enable bfloat16 mixed precision (requires Ampere+ GPU).

    Returns:
        Tuple of (trained_model, label_encoder, major_to_broad, metadata_dict).
    """
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments
    from sklearn.preprocessing import LabelEncoder

    from .faiss_retrieval import _load_jsonl

    num_iterations = num_iterations or cfg.train.setfit_num_iterations or 20
    num_epochs = num_epochs or cfg.train.setfit_num_epochs or 1

    # Build major→broad mapping
    taxonomy_path = cfg.resolve_path(cfg.paths.taxonomy_json, project_root)
    taxonomy = load_json(taxonomy_path)
    major_to_broad = {}
    for entry in taxonomy:
        major_to_broad.setdefault(entry["Major_Field_label"], entry["Broad_Field_label"])

    # Load training data
    if train_path is None:
        train_path = cfg.resolve_path(cfg.train.train_data, project_root)
    train_records = _load_jsonl(train_path)
    print(f"Loaded {len(train_records)} training abstracts")

    train_texts = [r["abstract"] for r in train_records]
    train_labels = [r["major_field"] for r in train_records]

    # Optionally subsample for speed
    if max_samples_per_class is not None:
        from collections import defaultdict
        by_class = defaultdict(list)
        for i, label in enumerate(train_labels):
            by_class[label].append(i)
        indices = []
        for label, idxs in by_class.items():
            indices.extend(idxs[:max_samples_per_class])
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        print(f"Subsampled to {len(train_texts)} training examples ({max_samples_per_class}/class)")

    # Encode labels to ints
    le = LabelEncoder()
    le.fit(train_labels)
    y_train = le.transform(train_labels).tolist()
    n_classes = len(le.classes_)

    # Build HF dataset
    train_ds = Dataset.from_dict({"text": train_texts, "label": y_train})

    # Pick encoder
    encoder_name = cfg.train.encoder or cfg.models.index_encoder
    print(f"SetFit encoder: {encoder_name}")
    print(f"Training: {num_iterations} iterations, {num_epochs} epochs, {n_classes} classes")

    # Create SetFit model
    model = SetFitModel.from_pretrained(encoder_name)

    # Reduce GPU memory: shorter sequences + gradient checkpointing
    if use_bf16:
        # Reduce max_seq_length (512 → 256; abstracts rarely exceed 256 tokens)
        model.model_body.max_seq_length = 256
        # Enable gradient checkpointing to trade compute for memory
        model.model_body[0].auto_model.gradient_checkpointing_enable()
        print(f"Memory optimization: max_seq_length=256, gradient_checkpointing=True")

    # Training arguments
    args = TrainingArguments(
        batch_size=batch_size,
        num_iterations=num_iterations,
        num_epochs=num_epochs,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
    )

    print("Training SetFit model...")
    trainer.train()
    print("Training complete.")

    # Save trained model
    model_dir = project_root / "output" / "models" / "setfit"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_dir))
    # Save label encoder classes
    import json
    with open(model_dir / "label_classes.json", "w") as f:
        json.dump(le.classes_.tolist(), f)
    print(f"Saved trained model to {model_dir}")

    metadata = {
        "encoder": encoder_name,
        "num_iterations": num_iterations,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "n_classes": n_classes,
        "n_train": len(train_records),
        "max_samples_per_class": max_samples_per_class,
    }

    return model, le, major_to_broad, metadata


def setfit_load(
    cfg: PipelineConfig,
    project_root: Path,
    model_dir: Path | None = None,
) -> tuple:
    """Load a previously trained SetFit model from disk.

    Returns (model, label_encoder, major_to_broad, metadata).
    """
    import json
    from setfit import SetFitModel
    from sklearn.preprocessing import LabelEncoder

    if model_dir is None:
        model_dir = project_root / "output" / "models" / "setfit"

    if not model_dir.exists():
        raise FileNotFoundError(f"No saved SetFit model found at {model_dir}. Train first.")

    print(f"Loading SetFit model from {model_dir}...")
    model = SetFitModel.from_pretrained(str(model_dir))

    # Load label encoder
    with open(model_dir / "label_classes.json") as f:
        classes = json.load(f)
    le = LabelEncoder()
    le.classes_ = np.array(classes)

    # Build major→broad mapping
    taxonomy_path = cfg.resolve_path(cfg.paths.taxonomy_json, project_root)
    taxonomy = load_json(taxonomy_path)
    major_to_broad = {}
    for entry in taxonomy:
        major_to_broad.setdefault(entry["Major_Field_label"], entry["Broad_Field_label"])

    metadata = {"encoder": "loaded_from_disk", "model_dir": str(model_dir)}
    print(f"Loaded model with {len(classes)} classes")
    return model, le, major_to_broad, metadata


def setfit_predict(
    model,
    le,
    major_to_broad: dict,
    test_records: list[dict],
    dataset_name: str = "synthetic_test",
    metadata: dict | None = None,
    num_iterations: int = 20,
    num_epochs: int = 1,
) -> "PredictionSet":
    """Predict on test records using a trained SetFit model.

    Args:
        model: Trained SetFitModel.
        le: Fitted LabelEncoder.
        major_to_broad: Major→broad mapping.
        test_records: List of dicts with 'abstract', 'major_field', 'broad_field'.
        dataset_name: Name for the prediction set.
        metadata: Optional metadata dict from training.
        num_iterations: Used for model naming.
        num_epochs: Used for model naming.

    Returns:
        PredictionSet with predictions.
    """
    from ..evaluation.predictions import Prediction, PredictionSet

    test_texts = [r["abstract"] for r in test_records]

    print(f"Predicting on {len(test_texts)} test abstracts...")
    preds = model.predict(test_texts)
    preds_np = np.array(preds)
    pred_labels = le.inverse_transform(preds_np)

    # Get probabilities if available
    try:
        probs = model.predict_proba(test_texts)
        probs_np = np.array(probs)
    except Exception:
        probs_np = None

    # Build predictions
    predictions = []
    for i in range(len(test_records)):
        predicted_major = pred_labels[i]
        predicted_broad = major_to_broad.get(predicted_major, "")

        if probs_np is not None:
            confidence = float(probs_np[i].max())
            top_k_idx = np.argsort(probs_np[i])[::-1][:10]
            top_k_fields = [le.classes_[idx] for idx in top_k_idx]
            top_k_scores = [float(probs_np[i][idx]) for idx in top_k_idx]
        else:
            confidence = 1.0
            top_k_fields = [predicted_major]
            top_k_scores = [1.0]

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

    model_name = f"setfit_{num_iterations}iter_{num_epochs}ep"
    meta = dict(metadata or {})
    meta["n_test"] = len(test_records)
    pred_set = PredictionSet(
        model_name=model_name,
        predictions=predictions,
        dataset=dataset_name,
        metadata=meta,
    )

    print(f"SetFit classification complete: {len(predictions)} predictions")
    return pred_set


def setfit_classify(
    cfg: PipelineConfig,
    project_root: Path,
    train_path: Path | None = None,
    test_path: Path | None = None,
    dataset_name: str = "synthetic_test",
    num_iterations: int | None = None,
    num_epochs: int | None = None,
    batch_size: int = 16,
    max_samples_per_class: int | None = None,
) -> "PredictionSet":
    """Train SetFit model on synthetic data, classify test set (convenience wrapper).

    For running on multiple test sets without re-training, use
    setfit_train() + setfit_predict() directly.
    """
    from .faiss_retrieval import _load_test_data

    model, le, major_to_broad, metadata = setfit_train(
        cfg, project_root, train_path=train_path,
        num_iterations=num_iterations, num_epochs=num_epochs,
        batch_size=batch_size, max_samples_per_class=max_samples_per_class,
    )

    # Load test data
    if test_path is None:
        test_path = cfg.resolve_path(cfg.train.test_data, project_root)
    test_records = _load_test_data(test_path, major_to_broad)
    print(f"Loaded {len(test_records)} test abstracts")

    return setfit_predict(
        model, le, major_to_broad, test_records,
        dataset_name=dataset_name, metadata=metadata,
        num_iterations=metadata["num_iterations"],
        num_epochs=metadata["num_epochs"],
    )
