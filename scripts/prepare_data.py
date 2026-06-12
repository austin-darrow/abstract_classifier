#!/usr/bin/env python
"""Stage 0: Parse CIP taxonomy and prepare data files.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --config configs/default.yaml configs/vista.yaml
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
    parser = argparse.ArgumentParser(description="Parse CIP taxonomy from Excel to JSON")
    parser.add_argument(
        "--config", "-c", nargs="+", type=Path,
        default=None,
        help="Config YAML file(s). Later files override earlier ones.",
    )
    args = parser.parse_args()

    project_root = find_project_root()

    config_paths = args.config
    if not config_paths:
        config_paths = [project_root / "configs" / "default.yaml"]

    cfg = load_config(*config_paths)

    from cip_classifier.data.taxonomy import run
    run(cfg, project_root)


if __name__ == "__main__":
    main()
