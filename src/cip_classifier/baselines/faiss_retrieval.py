"""Embedding-based retrieval classifier using FAISS.

Combines the build-index and classify steps into a single baseline module.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..utils import (
    build_faiss_index,
    encode_texts,
    load_faiss_index,
    load_json,
    load_model,
    save_faiss_index,
    save_json,
)


def build_index(cfg: PipelineConfig, project_root: Path) -> None:
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
            "Broad_Field_label": entry["Broad_Field_label"],
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
    embeddings = encode_texts(model, enriched_texts, batch_size=batch_size, mode="document")
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


def classify(cfg: PipelineConfig, project_root: Path) -> None:
    """Embed abstracts, query FAISS index, assign fields via majority vote."""
    device = cfg.get_device()
    print(f"Device: {device}")

    # Load index and metadata
    index_path = cfg.resolve_path(cfg.paths.faiss_index, project_root)
    metadata_path = cfg.resolve_path(cfg.paths.index_metadata, project_root)
    index = load_faiss_index(index_path)
    metadata = load_json(metadata_path)
    print(f"Index loaded: {index.ntotal} vectors")

    # Load model
    model = load_model(cfg.models.query_encoder, device)

    # Load valid major fields for validation
    major_fields_path = cfg.resolve_path(cfg.paths.major_fields_json, project_root)
    valid_major_fields = set(load_json(major_fields_path))

    # Load abstracts
    abstracts_path = cfg.resolve_path(cfg.paths.abstracts_excel, project_root)
    print(f"Loading abstracts from: {abstracts_path}")
    df = pd.read_excel(abstracts_path)
    abstracts = df["abstract"].fillna("").tolist()
    existing_labels = df["major_field"].tolist()
    print(f"Loaded {len(abstracts)} abstracts")

    # Embed abstracts
    batch_size = cfg.runtime.batch_size or cfg.classify.batch_size
    print(f"Encoding abstracts (batch_size={batch_size})...")
    embeddings = encode_texts(
        model, abstracts, batch_size=batch_size, prefix=cfg.models.query_prefix,
        mode="query",
    )

    # Save embeddings for downstream visualization
    embeddings_path = cfg.resolve_path(cfg.paths.embeddings_npy, project_root)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")

    # Query FAISS
    top_k = cfg.classify.top_k
    print(f"Querying top-{top_k} nearest neighbors...")
    similarities, indices = index.search(embeddings, top_k)

    # Aggregate results via majority vote
    results = []
    for i in range(len(abstracts)):
        top_k_indices = indices[i]
        top_k_sims = similarities[i]
        top_k_fields = [metadata[idx]["Major_Field_label"] for idx in top_k_indices]
        top_k_broad_fields = [metadata[idx]["Broad_Field_label"] for idx in top_k_indices]

        field_counts = Counter(top_k_fields)
        predicted_field, majority_count = field_counts.most_common(1)[0]

        broad_field_counts = Counter(top_k_broad_fields)
        predicted_broad_field, broad_majority_count = broad_field_counts.most_common(1)[0]

        results.append({
            "abstract": abstracts[i],
            "existing_label": existing_labels[i] if pd.notna(existing_labels[i]) else None,
            "predicted_field": predicted_field,
            "predicted_broad_field": predicted_broad_field,
            "top1_similarity": float(top_k_sims[0]),
            "agreement_ratio": majority_count / top_k,
            "broad_agreement_ratio": broad_majority_count / top_k,
            "top10_fields": top_k_fields,
            "top10_broad_fields": top_k_broad_fields,
            "top1_detailed_field": metadata[top_k_indices[0]]["Detailed_Field_label"],
            "top1_cip_title": metadata[top_k_indices[0]]["SED_CIPTitle"],
        })

    # Validate predictions
    invalid = [r for r in results if r["predicted_field"] not in valid_major_fields]
    if invalid:
        print(f"WARNING: {len(invalid)} predictions not in valid major fields set!")
    else:
        print("All predictions are valid major fields.")

    # Save results
    output_path = cfg.resolve_path(cfg.paths.classification_results, project_root)
    save_json(results, output_path)
    print(f"Saved {len(results)} results to {output_path}")

    # Summary stats
    has_label = [r for r in results if r["existing_label"] is not None]
    print(f"  Abstracts with existing labels: {len(has_label)}")
    print(f"  Mean top-1 similarity: {np.mean([r['top1_similarity'] for r in results]):.4f}")
    print(f"  Mean agreement ratio: {np.mean([r['agreement_ratio'] for r in results]):.2f}")
    print("Done.")


def run_all(cfg: PipelineConfig, project_root: Path) -> None:
    """Run the full baseline pipeline: parse → build-index → classify → evaluate."""
    from ..data.taxonomy import run as parse_run
    from ..evaluation.metrics import run as evaluate_run

    print("=" * 60)
    print("STEP 0: Parse taxonomy")
    print("=" * 60)
    parse_run(cfg, project_root)

    print("\n" + "=" * 60)
    print("STEP 1: Build FAISS index")
    print("=" * 60)
    build_index(cfg, project_root)

    print("\n" + "=" * 60)
    print("STEP 2: Classify abstracts")
    print("=" * 60)
    classify(cfg, project_root)

    print("\n" + "=" * 60)
    print("STEP 3: Evaluate results")
    print("=" * 60)
    evaluate_run(cfg, project_root)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


# ---------------------------------------------------------------------------
# B1: kNN on synthetic abstracts
# ---------------------------------------------------------------------------


def knn_classify(
    cfg: PipelineConfig,
    project_root: Path,
    train_path: Path | None = None,
    test_path: Path | None = None,
    dataset_name: str = "synthetic_test",
    top_k: int | None = None,
) -> "PredictionSet":
    """Embed synthetic training abstracts, build FAISS index, classify test set via kNN.

    Args:
        cfg: Pipeline config.
        project_root: Project root directory.
        train_path: Path to training JSONL. Defaults to config train_data.
        test_path: Path to test JSONL (or Excel for real TACC data).
        dataset_name: Name for the prediction set (e.g. "synthetic_test", "real_tacc").
        top_k: Number of neighbors for majority vote. Defaults to cfg.classify.top_k.

    Returns:
        PredictionSet with predictions for all test abstracts.
    """
    import json

    from ..evaluation.predictions import Prediction, PredictionSet

    device = cfg.get_device()
    k = top_k or cfg.classify.top_k

    # Load training data
    if train_path is None:
        train_path = cfg.resolve_path(cfg.train.train_data, project_root)
    train_records = _load_jsonl(train_path)
    print(f"Loaded {len(train_records)} training abstracts from {train_path}")

    # Load test data
    if test_path is None:
        test_path = cfg.resolve_path(cfg.train.test_data, project_root)
    test_records = _load_test_data(test_path)
    print(f"Loaded {len(test_records)} test abstracts from {test_path}")

    # Embed training abstracts
    model = load_model(cfg.models.index_encoder, device)
    batch_size = cfg.runtime.batch_size or cfg.index.batch_size

    train_texts = [r["abstract"] for r in train_records]
    train_labels = [r["major_field"] for r in train_records]
    train_broad = [r["broad_field"] for r in train_records]

    print(f"Encoding {len(train_texts)} training abstracts...")
    train_embeddings = encode_texts(model, train_texts, batch_size=batch_size, mode="document")

    # Build FAISS index
    index = build_faiss_index(train_embeddings)
    print(f"Built FAISS index: {index.ntotal} vectors")

    # Embed test abstracts
    test_texts = [r["abstract"] for r in test_records]
    print(f"Encoding {len(test_texts)} test abstracts...")
    test_embeddings = encode_texts(
        model, test_texts, batch_size=batch_size,
        prefix=cfg.models.query_prefix, mode="query",
    )

    # Query
    print(f"Querying top-{k} nearest neighbors...")
    similarities, indices = index.search(test_embeddings, k)

    # Build predictions via majority vote
    predictions = []
    for i in range(len(test_records)):
        top_k_indices = indices[i]
        top_k_sims = similarities[i]
        top_k_major = [train_labels[idx] for idx in top_k_indices]
        top_k_broad = [train_broad[idx] for idx in top_k_indices]

        # Majority vote for major field
        major_counts = Counter(top_k_major)
        predicted_major = major_counts.most_common(1)[0][0]

        # Majority vote for broad field
        broad_counts = Counter(top_k_broad)
        predicted_broad = broad_counts.most_common(1)[0][0]

        # Confidence = agreement ratio
        confidence = major_counts[predicted_major] / k

        # Top-k unique major fields by first occurrence
        seen = []
        scores = []
        for mf, sim in zip(top_k_major, top_k_sims):
            if mf not in seen:
                seen.append(mf)
                scores.append(float(sim))

        predictions.append(Prediction(
            abstract=test_records[i]["abstract"],
            true_major_field=test_records[i].get("major_field", ""),
            true_broad_field=test_records[i].get("broad_field", ""),
            predicted_major_field=predicted_major,
            predicted_broad_field=predicted_broad,
            confidence=confidence,
            top_k_major_fields=seen,
            top_k_scores=scores,
        ))

    encoder_name = cfg.models.index_encoder.split("/")[-1]
    pred_set = PredictionSet(
        model_name=f"knn_synth_{encoder_name}_k{k}",
        predictions=predictions,
        dataset=dataset_name,
        metadata={
            "encoder": cfg.models.index_encoder,
            "top_k": k,
            "n_train": len(train_records),
            "n_test": len(test_records),
        },
    )

    print(f"kNN classification complete: {len(predictions)} predictions")
    return pred_set


def _load_jsonl(path: Path) -> list[dict]:
    """Load records from a JSONL file."""
    import json
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _load_test_data(path: Path) -> list[dict]:
    """Load test data from JSONL or Excel (for real TACC abstracts)."""
    import json

    if path.suffix == ".jsonl":
        return _load_jsonl(path)
    elif path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
        records = []
        for _, row in df.iterrows():
            abstract = row.get("abstract", "")
            if pd.isna(abstract) or not abstract.strip():
                continue
            records.append({
                "abstract": str(abstract),
                "major_field": str(row.get("major_field", "")) if pd.notna(row.get("major_field")) else "",
                "broad_field": str(row.get("broad_field", "")) if pd.notna(row.get("broad_field")) else "",
            })
        return records
    else:
        raise ValueError(f"Unsupported test data format: {path.suffix}")
