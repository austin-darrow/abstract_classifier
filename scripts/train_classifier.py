#!/usr/bin/env python
"""Stage 2: Train a classifier on synthetic abstracts.

Usage:
    python scripts/train_classifier.py --config configs/train.yaml
    python scripts/train_classifier.py --config configs/train.yaml --model-type setfit
"""

import argparse
from pathlib import Path

from cip_classifier.config import load_config


def find_project_root() -> Path:
    """Walk up from CWD to find project root."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists() or (parent / "configs").is_dir():
            return parent
    return cwd


def main():
    parser = argparse.ArgumentParser(description="Train a field-of-science classifier")
    parser.add_argument(
        "--config", "-c", nargs="+", type=Path,
        default=None,
        help="Config YAML file(s). Later files override earlier ones.",
    )
    parser.add_argument("--model-type", type=str, default=None,
                        choices=["embedding_head", "setfit"],
                        help="Override model type from config")
    args = parser.parse_args()

    project_root = find_project_root()

    config_paths = args.config
    if not config_paths:
        config_paths = [
            project_root / "configs" / "default.yaml",
            project_root / "configs" / "train.yaml",
        ]

    cfg = load_config(*config_paths)

    if args.model_type is not None:
        cfg.train.model_type = args.model_type

    from cip_classifier.models.base import train_model
    train_model(cfg, project_root)


if __name__ == "__main__":
    main()
