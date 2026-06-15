#!/usr/bin/env python
"""B1: kNN classifier on synthetic abstracts.

Usage:
    python scripts/run_knn.py
    python scripts/run_knn.py --config configs/vista.yaml
    python scripts/run_knn.py --top-k 5
    python scripts/run_knn.py --test-data data/raw/database_abstracts.xlsx
"""

import argparse
from pathlib import Path

from cip_classifier.config import load_config


def find_project_root() -> Path:
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists() or (parent / "configs").is_dir():
            return parent
    return cwd


def main():
    parser = argparse.ArgumentParser(description="B1: kNN on synthetic abstracts")
    parser.add_argument(
        "--config", "-c", nargs="+", type=Path, default=None,
        help="Config YAML file(s).",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Number of neighbors.")
    parser.add_argument("--test-data", type=Path, default=None, help="Test data (JSONL or Excel).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    args = parser.parse_args()

    project_root = find_project_root()

    config_paths = args.config
    if not config_paths:
        default_cfg = project_root / "configs" / "default.yaml"
        config_paths = [default_cfg]
    cfg = load_config(*config_paths)

    output_dir = args.output_dir or project_root / "output" / "predictions"

    from cip_classifier.baselines.faiss_retrieval import knn_classify
    from cip_classifier.evaluation.metrics import compute_metrics, print_metrics
    from cip_classifier.evaluation.predictions import PredictionSet

    # --- Synthetic test ---
    print("=" * 60)
    print("B1: kNN on Synthetic Abstracts — Synthetic Test Set")
    print("=" * 60)
    pred_set = knn_classify(cfg, project_root, top_k=args.top_k, dataset_name="synthetic_test")
    metrics = compute_metrics(pred_set)
    print_metrics(metrics)
    pred_set.save(output_dir / f"predictions_{pred_set.model_name}_synthetic_test.json")
    metrics.save(output_dir / f"metrics_{pred_set.model_name}_synthetic_test.json")

    # --- Real TACC ---
    real_path = args.test_data or cfg.resolve_path(cfg.paths.abstracts_excel, project_root)
    if real_path.exists():
        print("\n" + "=" * 60)
        print("B1: kNN on Synthetic Abstracts — Real TACC Abstracts")
        print("=" * 60)
        pred_set_real = knn_classify(
            cfg, project_root, test_path=real_path,
            top_k=args.top_k, dataset_name="real_tacc",
        )
        # Filter to labeled records
        labeled = [p for p in pred_set_real.predictions if p.true_major_field]
        pred_set_labeled = PredictionSet(
            model_name=pred_set_real.model_name,
            predictions=labeled,
            dataset="real_tacc",
            metadata=pred_set_real.metadata,
        )
        metrics_real = compute_metrics(pred_set_labeled)
        print_metrics(metrics_real)
        pred_set_real.save(output_dir / f"predictions_{pred_set_real.model_name}_real_tacc.json")
        metrics_real.save(output_dir / f"metrics_{pred_set_real.model_name}_real_tacc.json")
    else:
        print(f"\nReal TACC data not found at {real_path}, skipping.")

    print("\nDone. Results saved to:", output_dir)


if __name__ == "__main__":
    main()
