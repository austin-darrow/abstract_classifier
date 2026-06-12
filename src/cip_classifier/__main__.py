"""CLI entry point for the CIP Classifier.

Usage:
    cip-classifier baseline -c configs/vista.yaml
    cip-classifier generate -c configs/generate.yaml
    cip-classifier train -c configs/train.yaml
    cip-classifier evaluate
"""

from __future__ import annotations

import functools
from pathlib import Path

import click

from .config import load_config


def _find_project_root() -> Path:
    """Walk up from CWD to find project root (contains pyproject.toml or configs/)."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists() or (parent / "configs").is_dir():
            return parent
    return cwd


def _config_options(fn):
    """Shared --config/-c option for all subcommands."""
    @click.option(
        "--config", "-c",
        "config_paths",
        multiple=True,
        type=click.Path(exists=True, path_type=Path),
        help="Config YAML file(s). Later files override earlier ones.",
    )
    @functools.wraps(fn)
    def wrapper(config_paths, *args, **kwargs):
        project_root = _find_project_root()

        if not config_paths:
            default_cfg = project_root / "configs" / "default.yaml"
            if default_cfg.exists():
                config_paths = (default_cfg,)
            else:
                click.echo("Error: No config file specified and configs/default.yaml not found.", err=True)
                raise SystemExit(1)

        cfg = load_config(*config_paths)
        return fn(cfg=cfg, project_root=project_root, *args, **kwargs)
    return wrapper


@click.group()
def cli() -> None:
    """CIP Classifier — classify research abstracts against CIP taxonomy."""


# ---------------------------------------------------------------------------
# Baseline pipeline (embedding retrieval)
# ---------------------------------------------------------------------------


@cli.command()
@_config_options
@click.option("--step", type=click.Choice(["parse", "build-index", "classify", "evaluate", "all"]),
              default="all", help="Run a specific step or the full pipeline.")
def baseline(cfg, project_root, step) -> None:
    """Run the embedding retrieval baseline (FAISS)."""
    from .baselines.faiss_retrieval import build_index, classify, run_all
    from .data.taxonomy import run as parse_run
    from .evaluation.metrics import run as evaluate_run

    if step == "all":
        run_all(cfg, project_root)
    elif step == "parse":
        parse_run(cfg, project_root)
    elif step == "build-index":
        build_index(cfg, project_root)
    elif step == "classify":
        classify(cfg, project_root)
    elif step == "evaluate":
        evaluate_run(cfg, project_root)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@cli.command()
@_config_options
@click.option("--samples", type=int, default=None, help="Override samples_per_cip.")
@click.option("--server-url", type=str, default=None, help="Override inference server URL.")
@click.option("--split-only", is_flag=True, help="Only split existing data (skip generation).")
def generate(cfg, project_root, samples, server_url, split_only) -> None:
    """Generate synthetic abstracts using LLM inference server."""
    if samples is not None:
        cfg.generate.samples_per_cip = samples
    if server_url is not None:
        cfg.generate.server_url = server_url

    if split_only:
        from .generation.pipeline import run_split
        run_split(cfg, project_root)
    else:
        from .generation.pipeline import run
        run(cfg, project_root)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@cli.command()
@_config_options
@click.option("--model-type", type=click.Choice(["embedding_head", "setfit"]),
              default=None, help="Override model type from config.")
def train(cfg, project_root, model_type) -> None:
    """Train a field-of-science classifier on synthetic data."""
    if model_type is not None:
        cfg.train.model_type = model_type

    from .models.base import train_model
    train_model(cfg, project_root)


# ---------------------------------------------------------------------------
# Evaluate (standalone)
# ---------------------------------------------------------------------------


@cli.command()
@_config_options
def evaluate(cfg, project_root) -> None:
    """Evaluate classification results."""
    from .evaluation.metrics import run
    run(cfg, project_root)


# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------


@cli.command()
@_config_options
@click.option("--method", type=click.Choice(["umap", "tsne"]), default=None,
              help="Override dimensionality reduction method.")
def visualize(cfg, project_root, method) -> None:
    """Visualize embedding space (UMAP or t-SNE)."""
    if method:
        cfg.visualize.method = method
    from .visualize import run
    run(cfg, project_root)


# ---------------------------------------------------------------------------
# Compare (multi-model evaluation)
# ---------------------------------------------------------------------------


@cli.command()
@_config_options
@click.option("--results-dir", type=click.Path(path_type=Path), default=None,
              help="Directory containing predictions_*.json files.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None,
              help="Directory to write comparison results.")
@click.option("--plot/--no-plot", default=True, help="Generate comparison plots.")
def compare(cfg, project_root, results_dir, output_dir, plot) -> None:
    """Compare multiple classifier approaches."""
    from .evaluation.comparison import run as compare_run, load_all_predictions, compare as compare_models
    from .evaluation.visualize import plot_comparison_bars, plot_confusion_matrix

    if results_dir is None:
        results_dir = project_root / "output" / "predictions"
    if output_dir is None:
        output_dir = project_root / "output" / "reports"

    compare_run(results_dir, output_dir)

    if plot:
        pred_sets = load_all_predictions(results_dir)
        if pred_sets:
            metrics_list = compare_models(pred_sets)
            plot_comparison_bars(
                metrics_list,
                output_dir / "comparison_chart.png",
            )
            # Confusion matrix for each model (broad field)
            for ps in pred_sets:
                safe_name = ps.model_name.replace("/", "_").replace(" ", "_")
                plot_confusion_matrix(
                    ps,
                    output_dir / f"confusion_{safe_name}.png",
                    level="broad",
                )


if __name__ == "__main__":
    cli()
