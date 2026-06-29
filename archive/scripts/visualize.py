#!/usr/bin/env python
"""Visualize embedding space with UMAP or t-SNE.

Usage:
    python scripts/visualize.py
    python scripts/visualize.py --method tsne
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
    parser = argparse.ArgumentParser(description="Visualize embedding space")
    parser.add_argument(
        "--config", "-c", nargs="+", type=Path,
        default=None,
        help="Config YAML file(s). Later files override earlier ones.",
    )
    parser.add_argument("--method", type=str, choices=["umap", "tsne"], default=None,
                        help="Override dimensionality reduction method.")
    args = parser.parse_args()

    project_root = find_project_root()

    config_paths = args.config
    if not config_paths:
        config_paths = [project_root / "configs" / "default.yaml"]

    cfg = load_config(*config_paths)

    if args.method:
        cfg.visualize.method = args.method

    from cip_classifier.visualize import run
    run(cfg, project_root)


if __name__ == "__main__":
    main()
