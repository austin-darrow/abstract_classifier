"""CLI entry point for the CIP Classifier pipeline.

Usage:
    python -m cip_classifier parse -c config/default.yaml
    python -m cip_classifier build-index -c config/default.yaml
    python -m cip_classifier classify -c config/default.yaml
    python -m cip_classifier evaluate -c config/default.yaml
    python -m cip_classifier run-all -c config/default.yaml

    # Override with Vista config:
    python -m cip_classifier classify -c config/default.yaml -c config/vista.yaml
"""

from __future__ import annotations

import functools
from pathlib import Path

import click

from .config import load_config


def _find_project_root() -> Path:
    """Walk up from CWD to find project root (contains pyproject.toml or config/)."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists() or (parent / "config").is_dir():
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
            default_cfg = project_root / "config" / "default.yaml"
            if default_cfg.exists():
                config_paths = (default_cfg,)
            else:
                click.echo("Error: No config file specified and config/default.yaml not found.", err=True)
                raise SystemExit(1)

        cfg = load_config(*config_paths)
        return fn(cfg=cfg, project_root=project_root, *args, **kwargs)
    return wrapper


@click.group()
def cli() -> None:
    """CIP Classifier — classify research abstracts against CIP taxonomy."""


@cli.command()
@_config_options
def parse(cfg, project_root) -> None:
    """Step 0: Parse CIP taxonomy from Excel into JSON files."""
    from .parse_fields import run
    run(cfg, project_root)


@cli.command("build-index")
@_config_options
def build_index(cfg, project_root) -> None:
    """Step 1: Build FAISS index from CIP taxonomy embeddings."""
    from .build_index import run
    run(cfg, project_root)


@cli.command()
@_config_options
def classify(cfg, project_root) -> None:
    """Step 2: Classify abstracts via nearest-neighbor matching."""
    from .classify import run
    run(cfg, project_root)


@cli.command()
@_config_options
def evaluate(cfg, project_root) -> None:
    """Step 3: Evaluate classification results against existing labels."""
    from .evaluate import run
    run(cfg, project_root)


@cli.command("run-all")
@_config_options
def run_all(cfg, project_root) -> None:
    """Run the full pipeline: parse → build-index → classify → evaluate."""
    from .parse_fields import run as parse_run
    from .build_index import run as index_run
    from .classify import run as classify_run
    from .evaluate import run as evaluate_run

    click.echo("=" * 60)
    click.echo("STEP 0: Parse taxonomy")
    click.echo("=" * 60)
    parse_run(cfg, project_root)

    click.echo("\n" + "=" * 60)
    click.echo("STEP 1: Build FAISS index")
    click.echo("=" * 60)
    index_run(cfg, project_root)

    click.echo("\n" + "=" * 60)
    click.echo("STEP 2: Classify abstracts")
    click.echo("=" * 60)
    classify_run(cfg, project_root)

    click.echo("\n" + "=" * 60)
    click.echo("STEP 3: Evaluate results")
    click.echo("=" * 60)
    evaluate_run(cfg, project_root)

    click.echo("\n" + "=" * 60)
    click.echo("PIPELINE COMPLETE")
    click.echo("=" * 60)


if __name__ == "__main__":
    cli()
