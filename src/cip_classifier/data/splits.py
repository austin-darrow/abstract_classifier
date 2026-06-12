"""Train/test/validation splitting utilities."""

from __future__ import annotations

import random
from pathlib import Path

from ..config import PipelineConfig
from ..generation.pipeline import load_jsonl, save_jsonl


def run_split(cfg: PipelineConfig, project_root: Path) -> None:
    """Split generated abstracts into train/test sets (stratified by field)."""
    from ..generation.pipeline import run_split as _run_split
    _run_split(cfg, project_root)
