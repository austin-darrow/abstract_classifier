#!/usr/bin/env python
"""Stage 1: Generate synthetic abstracts using LLM inference server.

Usage:
    python scripts/generate_abstracts.py --config configs/generate.yaml
    python scripts/generate_abstracts.py --config configs/generate.yaml --samples 5
    python scripts/generate_abstracts.py --split  # just split existing data
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
    parser = argparse.ArgumentParser(description="Generate synthetic abstracts via LLM")
    parser.add_argument(
        "--config", "-c", nargs="+", type=Path,
        default=None,
        help="Config YAML file(s). Later files override earlier ones.",
    )
    parser.add_argument("--samples", type=int, default=None, help="Override samples_per_cip")
    parser.add_argument("--server-url", type=str, default=None, help="Override server URL")
    parser.add_argument("--split", action="store_true", help="Only split existing data (no generation)")
    args = parser.parse_args()

    project_root = find_project_root()

    config_paths = args.config
    if not config_paths:
        config_paths = [
            project_root / "configs" / "default.yaml",
            project_root / "configs" / "generate.yaml",
        ]

    cfg = load_config(*config_paths)

    if args.samples is not None:
        cfg.generate.samples_per_cip = args.samples
    if args.server_url is not None:
        cfg.generate.server_url = args.server_url

    if args.split:
        from cip_classifier.generation.pipeline import run_split
        run_split(cfg, project_root)
    else:
        from cip_classifier.generation.pipeline import run
        run(cfg, project_root)


if __name__ == "__main__":
    main()
