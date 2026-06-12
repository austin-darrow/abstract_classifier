#!/usr/bin/env python
"""Stage 3: Run embedding retrieval baseline (build index + classify + evaluate).

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --config configs/default.yaml configs/vista.yaml
    python scripts/run_baseline.py --step classify
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
    parser = argparse.ArgumentParser(description="Run embedding retrieval baseline")
    parser.add_argument(
        "--config", "-c", nargs="+", type=Path,
        default=None,
        help="Config YAML file(s). Later files override earlier ones.",
    )
    parser.add_argument(
        "--step", type=str, default="all",
        choices=["parse", "build-index", "classify", "evaluate", "all"],
        help="Run a specific step or the full pipeline.",
    )
    args = parser.parse_args()

    project_root = find_project_root()

    config_paths = args.config
    if not config_paths:
        config_paths = [project_root / "configs" / "default.yaml"]

    cfg = load_config(*config_paths)

    from cip_classifier.baselines.faiss_retrieval import build_index, classify, run_all
    from cip_classifier.data.taxonomy import run as parse_run
    from cip_classifier.evaluation.metrics import run as evaluate_run

    if args.step == "all":
        run_all(cfg, project_root)
    elif args.step == "parse":
        parse_run(cfg, project_root)
    elif args.step == "build-index":
        build_index(cfg, project_root)
    elif args.step == "classify":
        classify(cfg, project_root)
    elif args.step == "evaluate":
        evaluate_run(cfg, project_root)


if __name__ == "__main__":
    main()
