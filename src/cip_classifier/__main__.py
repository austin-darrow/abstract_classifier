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
# B0 Baseline with evaluation framework
# ---------------------------------------------------------------------------


@cli.command("baseline-eval")
@_config_options
@click.option("--test-data", type=click.Path(exists=True, path_type=Path), default=None,
              help="Test data path (JSONL or Excel). Defaults to real TACC abstracts.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None,
              help="Directory to save predictions.")
def baseline_eval(cfg, project_root, test_data, output_dir) -> None:
    """Run B0 FAISS retrieval baseline with evaluation framework metrics."""
    from .baselines.faiss_retrieval import baseline_classify
    from .evaluation.metrics import compute_metrics, print_metrics
    from .evaluation.predictions import PredictionSet

    if output_dir is None:
        output_dir = project_root / "output" / "predictions"

    # Synthetic test
    synth_test = project_root / cfg.train.test_data
    if synth_test.exists():
        pred_set = baseline_classify(cfg, project_root, test_path=synth_test, dataset_name="synthetic_test")
        metrics = compute_metrics(pred_set)
        print_metrics(metrics)
        pred_set.save(output_dir / f"predictions_{pred_set.model_name}_synthetic_test.json")
        metrics.save(output_dir / f"metrics_{pred_set.model_name}_synthetic_test.json")

    # Real TACC
    real_path = test_data or cfg.resolve_path(cfg.paths.abstracts_excel, project_root)
    if real_path.exists():
        pred_set_real = baseline_classify(cfg, project_root, test_path=real_path, dataset_name="real_tacc")
        labeled = [p for p in pred_set_real.predictions
                   if p.true_major_field and p.true_major_field != "UNASSIGNED"]
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
# kNN baseline (B1)
# ---------------------------------------------------------------------------


@cli.command()
@_config_options
@click.option("--top-k", type=int, default=None, help="Number of neighbors for majority vote.")
@click.option("--test-data", type=click.Path(exists=True, path_type=Path), default=None,
              help="Test data path (JSONL or Excel). Defaults to synthetic test set.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None,
              help="Directory to save predictions.")
def knn(cfg, project_root, top_k, test_data, output_dir) -> None:
    """Run kNN classifier on synthetic training abstracts (B1)."""
    from .baselines.faiss_retrieval import knn_classify
    from .evaluation.metrics import compute_metrics, print_metrics

    if output_dir is None:
        output_dir = project_root / "output" / "predictions"

    # Synthetic test
    pred_set = knn_classify(cfg, project_root, top_k=top_k, dataset_name="synthetic_test")
    metrics = compute_metrics(pred_set)
    print_metrics(metrics)
    pred_set.save(output_dir / f"predictions_{pred_set.model_name}_synthetic_test.json")
    metrics.save(output_dir / f"metrics_{pred_set.model_name}_synthetic_test.json")

    # Real TACC
    real_path = test_data or cfg.resolve_path(cfg.paths.abstracts_excel, project_root)
    if real_path.exists():
        pred_set_real = knn_classify(
            cfg, project_root, test_path=real_path,
            top_k=top_k, dataset_name="real_tacc",
        )
        # Filter to only records with valid labels for metrics
        labeled = [p for p in pred_set_real.predictions
                   if p.true_major_field and p.true_major_field != "UNASSIGNED"]
        from .evaluation.predictions import PredictionSet
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


# ---------------------------------------------------------------------------
# TF-IDF + LogReg baseline (B2)
# ---------------------------------------------------------------------------


@cli.command()
@_config_options
@click.option("--test-data", type=click.Path(exists=True, path_type=Path), default=None,
              help="Test data path (JSONL or Excel). Defaults to synthetic test set.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None,
              help="Directory to save predictions.")
def tfidf(cfg, project_root, test_data, output_dir) -> None:
    """Run TF-IDF + Logistic Regression classifier (B2)."""
    from .baselines.tfidf_logreg import tfidf_classify
    from .evaluation.metrics import compute_metrics, print_metrics
    from .evaluation.predictions import PredictionSet

    if output_dir is None:
        output_dir = project_root / "output" / "predictions"

    # Synthetic test
    pred_set = tfidf_classify(cfg, project_root, dataset_name="synthetic_test")
    metrics = compute_metrics(pred_set)
    print_metrics(metrics)
    pred_set.save(output_dir / f"predictions_{pred_set.model_name}_synthetic_test.json")
    metrics.save(output_dir / f"metrics_{pred_set.model_name}_synthetic_test.json")

    # Real TACC
    real_path = test_data or cfg.resolve_path(cfg.paths.abstracts_excel, project_root)
    if real_path.exists():
        pred_set_real = tfidf_classify(
            cfg, project_root, test_path=real_path, dataset_name="real_tacc",
        )
        labeled = [p for p in pred_set_real.predictions
                   if p.true_major_field and p.true_major_field != "UNASSIGNED"]
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


# ---------------------------------------------------------------------------
# Embedding Head baseline (B3)
# ---------------------------------------------------------------------------


@cli.command()
@_config_options
@click.option("--test-data", type=click.Path(exists=True, path_type=Path), default=None,
              help="Test data path (JSONL or Excel). Defaults to synthetic test set.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None,
              help="Directory to save predictions.")
@click.option("--hidden-dim", type=int, default=None, help="MLP hidden dimension.")
@click.option("--epochs", type=int, default=None, help="Training epochs.")
@click.option("--lr", type=float, default=None, help="Learning rate.")
def embedding_head(cfg, project_root, test_data, output_dir, hidden_dim, epochs, lr) -> None:
    """Run frozen encoder + MLP head classifier (B3)."""
    from .baselines.embedding_head import embedding_head_classify
    from .evaluation.metrics import compute_metrics, print_metrics
    from .evaluation.predictions import PredictionSet

    if output_dir is None:
        output_dir = project_root / "output" / "predictions"

    kwargs = {}
    if hidden_dim is not None:
        kwargs["hidden_dim"] = hidden_dim
    if epochs is not None:
        kwargs["epochs"] = epochs
    if lr is not None:
        kwargs["lr"] = lr

    # Synthetic test
    pred_set = embedding_head_classify(cfg, project_root, dataset_name="synthetic_test", **kwargs)
    metrics = compute_metrics(pred_set)
    print_metrics(metrics)
    pred_set.save(output_dir / f"predictions_{pred_set.model_name}_synthetic_test.json")
    metrics.save(output_dir / f"metrics_{pred_set.model_name}_synthetic_test.json")

    # Real TACC
    real_path = test_data or cfg.resolve_path(cfg.paths.abstracts_excel, project_root)
    if real_path.exists():
        pred_set_real = embedding_head_classify(
            cfg, project_root, test_path=real_path, dataset_name="real_tacc", **kwargs,
        )
        labeled = [p for p in pred_set_real.predictions
                   if p.true_major_field and p.true_major_field != "UNASSIGNED"]
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


# ---------------------------------------------------------------------------
# SetFit baseline (B4)
# ---------------------------------------------------------------------------


@cli.command()
@_config_options
@click.option("--train-data", type=click.Path(exists=True, path_type=Path), default=None,
              help="Training data path (JSONL). Defaults to config train_data.")
@click.option("--test-data", type=click.Path(exists=True, path_type=Path), default=None,
              help="Test data path (JSONL or Excel). Defaults to synthetic test set.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None,
              help="Directory to save predictions.")
@click.option("--num-iterations", type=int, default=None, help="Contrastive training iterations.")
@click.option("--num-epochs", type=int, default=None, help="Fine-tuning epochs.")
@click.option("--batch-size", type=int, default=None, help="Contrastive pair batch size (default 16).")
@click.option("--max-samples-per-class", type=int, default=None,
              help="Cap training samples per class (speeds up training).")
@click.option("--bf16", is_flag=True, help="Enable bf16 mixed precision (halves GPU memory).")
@click.option("--predict-only", is_flag=True, help="Load saved model, skip training.")
def setfit(cfg, project_root, train_data, test_data, output_dir, num_iterations, num_epochs, batch_size, max_samples_per_class, bf16, predict_only) -> None:
    """Run SetFit contrastive fine-tuning classifier (B4)."""
    from .baselines.setfit_classify import setfit_train, setfit_predict, setfit_load
    from .baselines.faiss_retrieval import _load_test_data
    from .evaluation.metrics import compute_metrics, print_metrics
    from .evaluation.predictions import PredictionSet

    if output_dir is None:
        output_dir = project_root / "output" / "predictions"

    kwargs = {}
    if num_iterations is not None:
        kwargs["num_iterations"] = num_iterations
    if num_epochs is not None:
        kwargs["num_epochs"] = num_epochs
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if max_samples_per_class is not None:
        kwargs["max_samples_per_class"] = max_samples_per_class
    if bf16:
        kwargs["use_bf16"] = True

    # Train or load
    if predict_only:
        model, le, major_to_broad, metadata = setfit_load(cfg, project_root)
    else:
        model, le, major_to_broad, metadata = setfit_train(
            cfg, project_root, train_path=train_data, **kwargs
        )

    # Synthetic test
    synth_test_path = cfg.resolve_path(cfg.train.test_data, project_root)
    synth_records = _load_test_data(synth_test_path, major_to_broad)
    print(f"Loaded {len(synth_records)} synthetic test abstracts")
    n_iter = metadata.get("num_iterations", num_iterations or 20)
    n_ep = metadata.get("num_epochs", num_epochs or 1)
    pred_set = setfit_predict(
        model, le, major_to_broad, synth_records,
        dataset_name="synthetic_test", metadata=metadata,
        num_iterations=n_iter, num_epochs=n_ep,
    )
    metrics = compute_metrics(pred_set)
    print_metrics(metrics)
    pred_set.save(output_dir / f"predictions_{pred_set.model_name}_synthetic_test.json")
    metrics.save(output_dir / f"metrics_{pred_set.model_name}_synthetic_test.json")

    # Real TACC
    real_path = test_data or cfg.resolve_path(cfg.paths.abstracts_excel, project_root)
    if real_path.exists():
        real_records = _load_test_data(real_path, major_to_broad)
        print(f"Loaded {len(real_records)} real TACC abstracts")
        pred_set_real = setfit_predict(
            model, le, major_to_broad, real_records,
            dataset_name="real_tacc", metadata=metadata,
            num_iterations=n_iter, num_epochs=n_ep,
        )
        labeled = [p for p in pred_set_real.predictions
                   if p.true_major_field and p.true_major_field != "UNASSIGNED"]
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


# ---------------------------------------------------------------------------
# Full fine-tune (B5)
# ---------------------------------------------------------------------------


@cli.command()
@_config_options
@click.option("--train-data", type=click.Path(exists=True, path_type=Path), default=None,
              help="Training data path (JSONL). Defaults to config train_data.")
@click.option("--test-data", type=click.Path(exists=True, path_type=Path), default=None,
              help="Test data path (JSONL or Excel). Defaults to synthetic test set.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None,
              help="Directory to save predictions.")
@click.option("--model-name", type=str, default=None,
              help="HF model to fine-tune (default: allenai/scibert_scivocab_uncased).")
@click.option("--model-path", type=click.Path(exists=True, path_type=Path), default=None,
              help="Load a pre-trained model from this directory (skip training).")
@click.option("--epochs", type=int, default=3, help="Training epochs.")
@click.option("--lr", type=float, default=2e-5, help="Learning rate.")
@click.option("--batch-size", type=int, default=16, help="Per-device batch size.")
@click.option("--freeze-layers", type=int, default=0, help="Freeze first N encoder layers.")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility.")
def finetune(cfg, project_root, train_data, test_data, output_dir, model_name, model_path, epochs, lr, batch_size, freeze_layers, seed) -> None:
    """Run full encoder fine-tune classifier (B5)."""
    from .baselines.finetune import finetune_classify, finetune_predict
    from .evaluation.metrics import compute_metrics, print_metrics
    from .evaluation.predictions import PredictionSet

    if output_dir is None:
        output_dir = project_root / "output" / "predictions"

    if model_path is not None:
        # Predict-only mode: load saved model, predict on test sets
        pred_set = finetune_predict(
            cfg, project_root, model_path, dataset_name="synthetic_test",
        )
        metrics = compute_metrics(pred_set)
        print_metrics(metrics)
        pred_set.save(output_dir / f"predictions_{pred_set.model_name}_synthetic_test.json")
        metrics.save(output_dir / f"metrics_{pred_set.model_name}_synthetic_test.json")

        # Real TACC
        real_path = test_data or cfg.resolve_path(cfg.paths.abstracts_excel, project_root)
        if real_path.exists():
            pred_set_real = finetune_predict(
                cfg, project_root, model_path, test_path=real_path, dataset_name="real_tacc",
            )
            labeled = [p for p in pred_set_real.predictions
                       if p.true_major_field and p.true_major_field != "UNASSIGNED"]
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
        # Train + predict mode
        kwargs = {"epochs": epochs, "lr": lr, "batch_size": batch_size, "freeze_layers": freeze_layers, "seed": seed}
        if model_name is not None:
            kwargs["model_name_or_path"] = model_name
        if train_data is not None:
            kwargs["train_path"] = train_data

        # Synthetic test
        pred_set = finetune_classify(cfg, project_root, dataset_name="synthetic_test", **kwargs)
        metrics = compute_metrics(pred_set)
        print_metrics(metrics)
        pred_set.save(output_dir / f"predictions_{pred_set.model_name}_synthetic_test.json")
        metrics.save(output_dir / f"metrics_{pred_set.model_name}_synthetic_test.json")

        # Real TACC
        real_path = test_data or cfg.resolve_path(cfg.paths.abstracts_excel, project_root)
        if real_path.exists():
            pred_set_real = finetune_classify(
                cfg, project_root, test_path=real_path, dataset_name="real_tacc", **kwargs,
            )
            labeled = [p for p in pred_set_real.predictions
                       if p.true_major_field and p.true_major_field != "UNASSIGNED"]
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


# ---------------------------------------------------------------------------
# Zero-shot LLM (B6)
# ---------------------------------------------------------------------------


@cli.command()
@_config_options
@click.option("--test-data", type=click.Path(exists=True, path_type=Path), default=None,
              help="Test data path (JSONL or Excel). Defaults to synthetic test set.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None,
              help="Directory to save predictions.")
@click.option("--model-name", type=str, default=None, help="LLM model name.")
@click.option("--server-url", type=str, default=None, help="OpenAI-compatible API URL.")
@click.option("--max-samples", type=int, default=None, help="Cap samples for cost control.")
@click.option("--concurrency", type=int, default=8, help="Concurrent API requests.")
def zeroshot(cfg, project_root, test_data, output_dir, model_name, server_url, max_samples, concurrency) -> None:
    """Run zero-shot LLM classifier (B6)."""
    from .baselines.zeroshot_llm import zeroshot_llm_classify
    from .evaluation.metrics import compute_metrics, print_metrics
    from .evaluation.predictions import PredictionSet

    if output_dir is None:
        output_dir = project_root / "output" / "predictions"

    kwargs = {"concurrency": concurrency}
    if model_name is not None:
        kwargs["model"] = model_name
    if server_url is not None:
        kwargs["server_url"] = server_url
    if max_samples is not None:
        kwargs["max_samples"] = max_samples

    # Synthetic test
    if test_data:
        test_path = test_data
    else:
        test_path = None

    pred_set = zeroshot_llm_classify(
        cfg, project_root, test_path=test_path, dataset_name="synthetic_test" if not test_data else "real_tacc",
        **kwargs,
    )
    metrics = compute_metrics(pred_set)
    print_metrics(metrics)
    pred_set.save(output_dir / f"predictions_{pred_set.model_name}_{pred_set.dataset}.json")
    metrics.save(output_dir / f"metrics_{pred_set.model_name}_{pred_set.dataset}.json")

    # If no explicit test data, also run on real TACC
    if not test_data:
        real_path = cfg.resolve_path(cfg.paths.abstracts_excel, project_root)
        if real_path.exists():
            pred_set_real = zeroshot_llm_classify(
                cfg, project_root, test_path=real_path, dataset_name="real_tacc", **kwargs,
            )
            labeled = [p for p in pred_set_real.predictions
                   if p.true_major_field and p.true_major_field != "UNASSIGNED"]
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
