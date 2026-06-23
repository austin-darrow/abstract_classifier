"""Base interface and dispatcher for classifier models."""

from __future__ import annotations

from pathlib import Path

from ..config import PipelineConfig


def train_model(cfg: PipelineConfig, project_root: Path) -> None:
    """Train a classifier based on config model_type."""
    model_type = cfg.train.model_type

    if model_type == "setfit":
        from .setfit_model import train
        train(cfg, project_root)
    elif model_type == "finetune":
        from ..baselines.finetune import train
        train(cfg, project_root)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'setfit' or 'finetune'.")
