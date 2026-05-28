"""Step 1: Build FAISS index from CIP taxonomy embeddings."""

from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .utils import (
    build_faiss_index,
    encode_texts,
    load_json,
    load_model,
    save_faiss_index,
    save_json,
)


def run(cfg: PipelineConfig, project_root: Path) -> None:
    """Load taxonomy, embed entries, build and save FAISS index."""
    taxonomy_path = cfg.resolve_path(cfg.paths.taxonomy_json, project_root)
    cip_entries = load_json(taxonomy_path)
    print(f"Loaded {len(cip_entries)} CIP entries from {taxonomy_path}")

    # Build enriched text and metadata
    enriched_texts = []
    metadata = []

    for entry in cip_entries:
        text = cfg.index.text_template.format(**entry)
        enriched_texts.append(text)
        metadata.append({
            "Major_Field_label": entry["Major_Field_label"],
            "Detailed_Field_label": entry["Detailed_Field_label"],
            "SED_CIPTitle": entry["SED_CIPTitle"],
        })

    print(f"Built {len(enriched_texts)} enriched text strings")
    print(f"Sample: {enriched_texts[0][:200]}...")

    # Embed taxonomy
    device = cfg.get_device()
    model = load_model(cfg.models.index_encoder, device)

    batch_size = cfg.runtime.batch_size or cfg.index.batch_size
    print(f"Encoding taxonomy entries (batch_size={batch_size})...")
    embeddings = encode_texts(model, enriched_texts, batch_size=batch_size)
    print(f"Embeddings shape: {embeddings.shape}")

    # Build and save FAISS index
    index = build_faiss_index(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dimension {embeddings.shape[1]}")

    index_path = cfg.resolve_path(cfg.paths.faiss_index, project_root)
    save_faiss_index(index, index_path)

    metadata_path = cfg.resolve_path(cfg.paths.index_metadata, project_root)
    save_json(metadata, metadata_path)

    print(f"Saved: {index_path}, {metadata_path}")
    print("Done.")
