"""SetFit-based few-shot classifier."""

from __future__ import annotations

from pathlib import Path

from ..config import PipelineConfig


def train(cfg: PipelineConfig, project_root: Path) -> None:
    """Train a SetFit model for field-of-science classification."""
    raise NotImplementedError(
        "SetFit training not yet implemented. "
        "This will be implemented in Phase 2 of the project."
    )
