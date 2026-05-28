"""CLI entry point for the CIP Classifier pipeline.

Usage:
    python -m cip_classifier parse --config config/default.yaml
    python -m cip_classifier build-index --config config/default.yaml
    python -m cip_classifier classify --config config/default.yaml
    python -m cip_classifier evaluate --config config/default.yaml
    python -m cip_classifier run-all --config config/default.yaml

    # Override with Vista config:
    python -m cip_classifier classify --config config/default.yaml --config config/vista.yaml
"""

from __future__ import annotations

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


@click.group()
@click.option(
    "--config", "-c",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Config YAML file(s). Later files override earlier ones.",
)
@click.pass_context
def cli(ctx: click.Context, config: tuple[Path, ...]) -> None:
    """CIP Classifier — classify research abstracts against CIP taxonomy."""
    ctx.ensure_object(dict)
    project_root = _find_project_root()
    ctx.obj["project_root"] = project_root

    if not config:
        # Default to config/default.yaml if it exists
        default_cfg = project_root / "config" / "default.yaml"
        if default_cfg.exists():
            config = (default_cfg,)
        else:
            click.echo("Error: No config file specified and config/default.yaml not found.", err=True)
            raise SystemExit(1)

    ctx.obj["cfg"] = load_config(*config)


@cli.command()
@click.pass_context
def parse(ctx: click.Context) -> None:
    """Step 0: Parse CIP taxonomy from Excel into JSON files."""
    from .parse_fields import run
    run(ctx.obj["cfg"], ctx.obj["project_root"])


@cli.command("build-index")
@click.pass_context
def build_index(ctx: click.Context) -> None:
    """Step 1: Build FAISS index from CIP taxonomy embeddings."""
    from .build_index import run
    run(ctx.obj["cfg"], ctx.obj["project_root"])


@cli.command()
@click.pass_context
def classify(ctx: click.Context) -> None:
    """Step 2: Classify abstracts via nearest-neighbor matching."""
    from .classify import run
    run(ctx.obj["cfg"], ctx.obj["project_root"])


@cli.command()
@click.pass_context
def evaluate(ctx: click.Context) -> None:
    """Step 3: Evaluate classification results against existing labels."""
    from .evaluate import run
    run(ctx.obj["cfg"], ctx.obj["project_root"])


@cli.command("run-all")
@click.pass_context
def run_all(ctx: click.Context) -> None:
    """Run the full pipeline: parse → build-index → classify → evaluate."""
    cfg = ctx.obj["cfg"]
    root = ctx.obj["project_root"]

    from .parse_fields import run as parse_run
    from .build_index import run as index_run
    from .classify import run as classify_run
    from .evaluate import run as evaluate_run

    click.echo("=" * 60)
    click.echo("STEP 0: Parse taxonomy")
    click.echo("=" * 60)
    parse_run(cfg, root)

    click.echo("\n" + "=" * 60)
    click.echo("STEP 1: Build FAISS index")
    click.echo("=" * 60)
    index_run(cfg, root)

    click.echo("\n" + "=" * 60)
    click.echo("STEP 2: Classify abstracts")
    click.echo("=" * 60)
    classify_run(cfg, root)

    click.echo("\n" + "=" * 60)
    click.echo("STEP 3: Evaluate results")
    click.echo("=" * 60)
    evaluate_run(cfg, root)

    click.echo("\n" + "=" * 60)
    click.echo("PIPELINE COMPLETE")
    click.echo("=" * 60)


if __name__ == "__main__":
    cli()
