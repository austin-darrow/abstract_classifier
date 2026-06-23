"""B3: Frozen encoder + MLP classification head.

Uses pre-computed embeddings from bge-large-en-v1.5, trains a small MLP on top.
Fast to train, tests whether a learned decision boundary helps over kNN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import PipelineConfig
from ..utils import load_json


class MLPHead(nn.Module):
    """Two-layer MLP classification head."""

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def embedding_head_classify(
    cfg: PipelineConfig,
    project_root: Path,
    train_path: Path | None = None,
    test_path: Path | None = None,
    dataset_name: str = "synthetic_test",
    hidden_dim: int | None = None,
    dropout: float | None = None,
    lr: float | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> "PredictionSet":
    """Train frozen-encoder + MLP head, classify test set.

    Args:
        cfg: Pipeline config.
        project_root: Project root directory.
        train_path: Path to training JSONL.
        test_path: Path to test JSONL or Excel.
        dataset_name: Name for the prediction set.
        hidden_dim: MLP hidden dimension.
        dropout: Dropout rate.
        lr: Learning rate.
        epochs: Number of training epochs.
        batch_size: Training batch size.

    Returns:
        PredictionSet with predictions for all test abstracts.
    """
    from sklearn.preprocessing import LabelEncoder as SkLabelEncoder

    from ..evaluation.predictions import Prediction, PredictionSet
    from ..utils import encode_texts, load_model
    from .faiss_retrieval import _load_jsonl, _load_test_data

    # Hyperparams (config overrides → function args → defaults)
    hidden_dim = hidden_dim or cfg.train.hidden_dim or 256
    dropout = dropout if dropout is not None else cfg.train.dropout
    lr = lr or cfg.train.learning_rate or 1e-3
    epochs = epochs or cfg.train.epochs or 20
    batch_size = batch_size or cfg.train.batch_size or 64

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
    le = SkLabelEncoder()
    le.fit(train_labels)
    y_train = le.transform(train_labels)
    n_classes = len(le.classes_)

    # Encode text with frozen encoder
    device = cfg.get_device()
    model = load_model(cfg.models.index_encoder, device)
    enc_batch_size = cfg.runtime.batch_size or cfg.index.batch_size

    print(f"Encoding {len(train_texts)} train abstracts (frozen encoder)...")
    X_train = encode_texts(model, train_texts, batch_size=enc_batch_size, mode="document")
    print(f"Encoding {len(test_texts)} test abstracts...")
    X_test = encode_texts(model, test_texts, batch_size=enc_batch_size, mode="query")

    input_dim = X_train.shape[1]
    print(f"Embedding dim: {input_dim}, n_classes: {n_classes}")

    # Move to torch
    torch_device = torch.device(device if device != "auto" else "cpu")
    X_train_t = torch.from_numpy(X_train).float().to(torch_device)
    y_train_t = torch.from_numpy(y_train).long().to(torch_device)
    X_test_t = torch.from_numpy(X_test).float().to(torch_device)

    # Train MLP
    mlp = MLPHead(input_dim, hidden_dim, n_classes, dropout).to(torch_device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    print(f"Training MLP (hidden={hidden_dim}, epochs={epochs}, lr={lr})...")
    mlp.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            logits = mlp(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} — loss: {total_loss/total:.4f}, acc: {correct/total:.4f}")

    # Predict
    mlp.eval()
    with torch.no_grad():
        logits = mlp(X_test_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    y_pred = probs.argmax(axis=1)
    pred_labels = le.inverse_transform(y_pred)

    # Build predictions
    predictions = []
    for i in range(len(test_records)):
        predicted_major = pred_labels[i]
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

    model_name = f"embedding_head_{hidden_dim}h_{epochs}ep"
    pred_set = PredictionSet(
        model_name=model_name,
        predictions=predictions,
        dataset=dataset_name,
        metadata={
            "encoder": cfg.models.index_encoder,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "input_dim": input_dim,
            "n_classes": n_classes,
            "n_train": len(train_records),
            "n_test": len(test_records),
        },
    )

    print(f"Embedding head classification complete: {len(predictions)} predictions")
    return pred_set
