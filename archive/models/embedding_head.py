"""Embedding head classifier: frozen encoder + trained MLP/linear head."""

from __future__ import annotations

from pathlib import Path

from ..config import PipelineConfig


def train(cfg: PipelineConfig, project_root: Path) -> None:
    """Train an MLP classification head on frozen embeddings."""
    raise NotImplementedError(
        "Embedding head training not yet implemented. "
        "This will be implemented in Phase 2 of the project."
    )
