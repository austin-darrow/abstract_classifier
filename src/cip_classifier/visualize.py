"""Step 4 (optional): Visualize embedding space with UMAP or t-SNE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .utils import encode_texts, load_json, load_model


def _build_broad_field_lookup(taxonomy: list[dict]) -> dict[str, str]:
    """Build a mapping from Major_Field_label → Broad_Field_label."""
    lookup = {}
    for entry in taxonomy:
        major = entry["Major_Field_label"]
        broad = entry["Broad_Field_label"]
        if major not in lookup:
            lookup[major] = broad
    return lookup


def run(cfg: PipelineConfig, project_root: Path) -> None:
    """Generate 2D embedding visualization colored by broad field."""
    # Load classification results for labels
    results_path = cfg.resolve_path(cfg.paths.classification_results, project_root)
    results = load_json(results_path)
    print(f"Loaded {len(results)} classification results")

    # Build major → broad field mapping
    taxonomy_path = cfg.resolve_path(cfg.paths.taxonomy_json, project_root)
    taxonomy = load_json(taxonomy_path)
    major_to_broad = _build_broad_field_lookup(taxonomy)

    # Get broad field labels for each abstract
    broad_fields = []
    for r in results:
        major = r["predicted_field"]
        broad = major_to_broad.get(major, "Unknown")
        broad_fields.append(broad)

    # Load or compute embeddings
    embeddings_path = cfg.resolve_path(cfg.paths.embeddings_npy, project_root)
    if embeddings_path.exists():
        print(f"Loading saved embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
    else:
        print("No saved embeddings found, re-encoding abstracts...")
        device = cfg.get_device()
        model = load_model(cfg.models.query_encoder, device)

        abstracts_path = cfg.resolve_path(cfg.paths.abstracts_excel, project_root)
        df = pd.read_excel(abstracts_path)
        abstracts = df["abstract"].fillna("").tolist()

        batch_size = cfg.runtime.batch_size or cfg.classify.batch_size
        embeddings = encode_texts(
            model, abstracts, batch_size=batch_size,
            prefix=cfg.models.query_prefix, mode="query",
        )

    print(f"Embeddings shape: {embeddings.shape}")

    # Dimensionality reduction
    method = cfg.visualize.method
    print(f"Reducing to 2D with {method.upper()}...")

    if method == "umap":
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=cfg.visualize.n_neighbors,
            min_dist=cfg.visualize.min_dist,
            metric="cosine",
            random_state=42,
        )
        coords = reducer.fit_transform(embeddings)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2,
            perplexity=cfg.visualize.perplexity,
            metric="cosine",
            random_state=42,
            init="pca",
        )
        coords = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown visualization method: {method}. Use 'umap' or 'tsne'.")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique_fields = sorted(set(broad_fields))
    n_colors = len(unique_fields)
    cmap = plt.cm.get_cmap("tab20", max(n_colors, 20))
    # Extend colormap if more than 20 categories
    if n_colors > 20:
        cmap = plt.cm.get_cmap("gist_ncar", n_colors)

    field_to_color = {field: cmap(i) for i, field in enumerate(unique_fields)}

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    for field in unique_fields:
        mask = [f == field for f in broad_fields]
        indices = np.where(mask)[0]
        ax.scatter(
            coords[indices, 0],
            coords[indices, 1],
            c=[field_to_color[field]],
            label=field,
            s=8,
            alpha=0.6,
        )

    ax.set_title(f"Abstract Embeddings ({method.upper()}) — Colored by Broad Field (N={n_colors})")
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=7,
        markerscale=2,
        frameon=False,
    )
    plt.tight_layout()

    output_path = cfg.resolve_path(cfg.paths.visualization_plot, project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")
