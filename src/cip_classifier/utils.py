"""Shared utilities for model loading, embedding, and FAISS I/O."""

from __future__ import annotations

from pathlib import Path

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer


def load_model(model_name: str, device: str) -> SentenceTransformer:
    """Load a SentenceTransformer model on the specified device."""
    print(f"Loading model: {model_name} (device={device})")
    return SentenceTransformer(model_name, device=device)


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    prefix: str = "",
    show_progress: bool = True,
) -> np.ndarray:
    """Encode a list of texts into normalized embeddings.

    Args:
        model: Loaded SentenceTransformer.
        texts: Raw text strings to embed.
        batch_size: Encoding batch size.
        prefix: Optional prefix prepended to each text before encoding.
        show_progress: Whether to show a progress bar.

    Returns:
        Normalized embeddings as float32 numpy array of shape (N, dim).
    """
    if prefix:
        texts = [prefix + t for t in texts]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        batch_size=batch_size,
    )
    return np.array(embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from normalized embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def save_faiss_index(index: faiss.IndexFlatIP, path: Path) -> None:
    """Write a FAISS index to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_faiss_index(path: Path) -> faiss.IndexFlatIP:
    """Load a FAISS index from disk."""
    return faiss.read_index(str(path))


def save_json(data, path: Path) -> None:
    """Write data as JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path):
    """Load and parse a JSON file."""
    with open(path) as f:
        return json.load(f)
