"""D2.2: Generate detailed-level silver labels from hierarchical model on real TACC.

Runs the hierarchical model (Strategy C) on all real TACC abstracts and outputs
pseudo-labeled records with detailed_field annotations at high confidence.

Output: data/generated/detailed_silver_labels.jsonl
Does NOT overwrite existing silver_labels.jsonl (which has major-field only).

Usage:
    # Run on Vista (GPU required):
    python scripts/generate_detailed_silver.py

    # Custom threshold:
    python scripts/generate_detailed_silver.py --threshold 0.85

    # Use specific models:
    python scripts/generate_detailed_silver.py \
        --major-model output/sweep/models/scibert_...freeze8 \
        --detailed-model output/models/detailed_finetune
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def find_project_root() -> Path:
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return cwd


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {path}")


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def find_major_model(project_root: Path) -> Path | None:
    """Auto-discover major-field model (same logic as run_hierarchical_inference.py)."""
    for search_dir in [
        project_root / "output" / "sweep" / "models",
        project_root / "output" / "models" / "finetune",
    ]:
        if not search_dir.exists():
            continue
        for d in sorted(search_dir.iterdir()):
            lc = d / "label_classes.json"
            if lc.exists():
                with open(lc) as f:
                    classes = json.load(f)
                if 60 <= len(classes) <= 80:
                    return d
    return None


def load_real_tacc_abstracts(project_root: Path) -> list[dict]:
    """Load real TACC abstracts from predictions or raw data."""
    # Try loading from existing prediction file (has abstracts)
    pred_dir = project_root / "output" / "predictions"
    for p in sorted(pred_dir.glob("predictions_*real_tacc.json")):
        with open(p) as f:
            data = json.load(f)
        records = data.get("predictions", [])
        if records:
            print(f"Loaded {len(records)} real TACC abstracts from {p.name}")
            return records

    # Fallback: load from raw Excel
    print("No prediction files found. Loading from raw data...")
    from cip_classifier.config import load_config
    cfg = load_config(project_root / "configs" / "default.yaml")
    import pandas as pd
    df = pd.read_excel(project_root / cfg.paths.abstracts_excel)
    records = []
    for _, row in df.iterrows():
        abstract = str(row.get("Abstract", "")).strip()
        if abstract and abstract != "nan":
            records.append({
                "abstract": abstract,
                "db_major_field": str(row.get("major_field", "")),
                "db_broad_field": str(row.get("broad_field", "")),
            })
    print(f"Loaded {len(records)} abstracts from Excel")
    return records


def main():
    parser = argparse.ArgumentParser(description="D2.2: Generate detailed silver labels from hierarchical model")
    parser.add_argument("--major-model", type=Path, default=None, help="Major-field model directory")
    parser.add_argument("--detailed-model", type=Path, default=None, help="Detailed-field model directory")
    parser.add_argument("--threshold", type=float, default=0.80, help="Confidence threshold for silver labels (default: 0.80)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--top-k-major", type=int, default=5, help="Top-k major fields for Strategy C")
    parser.add_argument("--stats-only", action="store_true", help="Show stats from existing output without re-running")
    args = parser.parse_args()

    project_root = find_project_root()
    output_path = args.output or (project_root / "data" / "generated" / "detailed_silver_labels.jsonl")
    stats_path = output_path.with_suffix(".stats.json")

    if args.stats_only:
        if output_path.exists():
            records = load_jsonl(output_path)
            print(f"Detailed silver labels: {len(records)}")
            field_counts = Counter(r.get("major_field", "") for r in records)
            for field, n in field_counts.most_common(20):
                print(f"  {field}: {n}")
            if stats_path.exists():
                with open(stats_path) as f:
                    print(json.dumps(json.load(f), indent=2))
        else:
            print(f"No output found at {output_path}")
        return

    # Import heavy dependencies only when needed
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    def _load_model_and_predict(model_dir, texts, batch_size, max_length=512):
        """Load a saved model and return raw logits."""
        with open(model_dir / "label_classes.json") as f:
            label_classes = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        def tokenize_fn(examples):
            return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")

        ds = Dataset.from_dict({"text": texts}).map(tokenize_fn, batched=True)
        ds.set_format("torch", columns=["input_ids", "attention_mask"])

        all_logits = []
        with torch.no_grad():
            for i in range(0, len(ds), batch_size):
                batch = ds[i:i + batch_size]
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(outputs.logits.cpu().numpy())
        return np.concatenate(all_logits, axis=0), label_classes

    def _build_taxonomy_mappings(taxonomy_path):
        """Build detailed→major, detailed→broad, and major→set(detailed) mappings."""
        with open(taxonomy_path) as f:
            taxonomy = json.load(f)
        d2m, d2b = {}, {}
        m2d: dict[str, set] = {}
        for entry in taxonomy:
            d = entry["Detailed_Field_label"]
            m = entry["Major_Field_label"]
            b = entry["Broad_Field_label"]
            d2m.setdefault(d, m)
            d2b.setdefault(d, b)
            m2d.setdefault(m, set()).add(d)
        return d2m, d2b, m2d

    def _build_major_to_detailed_indices(major_classes, detailed_classes, major_to_detailed_set):
        """For each major class index, find which detailed class indices belong to it."""
        result = {}
        for mi, major in enumerate(major_classes):
            detailed_set = major_to_detailed_set.get(major, set())
            result[mi] = [di for di, d in enumerate(detailed_classes) if d in detailed_set]
        return result

    # Resolve model paths
    major_model = args.major_model
    if major_model is None:
        major_model = find_major_model(project_root)
        if major_model is None:
            print("ERROR: Could not find major-field model. Use --major-model.")
            return
    print(f"Major model: {major_model}")

    detailed_model = args.detailed_model or (project_root / "output" / "models" / "detailed_finetune")
    print(f"Detailed model: {detailed_model}")

    # Load taxonomy
    taxonomy_path = project_root / "data" / "processed" / "SEDCIP24.json"
    detailed_to_major, detailed_to_broad, major_to_detailed_set = _build_taxonomy_mappings(taxonomy_path)

    # Load real TACC abstracts
    real_records = load_real_tacc_abstracts(project_root)
    texts = [r["abstract"] for r in real_records]
    print(f"\nRunning inference on {len(texts)} real TACC abstracts...")

    # Get logits from both models
    print("Loading major model...")
    major_logits, major_classes = _load_model_and_predict(major_model, texts, args.batch_size)
    print(f"  {len(major_classes)} major classes")

    print("Loading detailed model...")
    detailed_logits, detailed_classes = _load_model_and_predict(detailed_model, texts, args.batch_size)
    print(f"  {len(detailed_classes)} detailed classes")

    # Build index mapping
    major_to_detailed_idx = _build_major_to_detailed_indices(
        major_classes, detailed_classes, major_to_detailed_set,
    )

    # Apply Strategy C (combined = major_prob × detailed_prob)
    print(f"\nApplying Strategy C with threshold={args.threshold}...")
    major_probs = softmax(major_logits)
    detailed_probs = softmax(detailed_logits)

    silver_labels = []
    confidence_all = []
    n_above_threshold = 0

    for i in range(len(texts)):
        # Get top-k major predictions
        top_major_idx = np.argsort(major_probs[i])[::-1][:args.top_k_major]

        best_score = -1.0
        best_detailed_name = None
        best_major_name = None
        best_broad_name = None
        best_major_conf = 0.0

        for mi in top_major_idx:
            major_name = major_classes[mi]
            major_prob = major_probs[i, mi]
            valid_detailed = major_to_detailed_idx.get(int(mi), [])
            if not valid_detailed:
                continue

            for di in valid_detailed:
                score = float(major_prob * detailed_probs[i, di])
                if score > best_score:
                    best_score = score
                    best_detailed_name = detailed_classes[di]
                    best_major_name = major_name
                    best_broad_name = detailed_to_broad.get(detailed_classes[di], "")
                    best_major_conf = float(major_prob)

        if best_detailed_name is None:
            continue

        confidence_all.append(best_score)

        if best_score >= args.threshold:
            n_above_threshold += 1
            record = {
                "abstract": texts[i],
                "detailed_field": best_detailed_name,
                "major_field": best_major_name,
                "broad_field": best_broad_name,
                "source": "detailed_silver",
                "confidence": round(best_score, 6),
                "major_confidence": round(best_major_conf, 6),
            }
            # Carry over DB labels if available
            if "true_major_field" in real_records[i]:
                record["db_major_field"] = real_records[i]["true_major_field"]
                record["db_broad_field"] = real_records[i].get("true_broad_field", "")
            elif "db_major_field" in real_records[i]:
                record["db_major_field"] = real_records[i]["db_major_field"]
                record["db_broad_field"] = real_records[i].get("db_broad_field", "")
            silver_labels.append(record)

    # Save
    save_jsonl(silver_labels, output_path)

    # Stats
    conf_array = np.array(confidence_all)
    stats = {
        "total_abstracts": len(texts),
        "threshold": args.threshold,
        "above_threshold": n_above_threshold,
        "yield_pct": round(n_above_threshold / len(texts) * 100, 1),
        "confidence_mean": round(float(conf_array.mean()), 4),
        "confidence_median": round(float(np.median(conf_array)), 4),
        "confidence_p25": round(float(np.percentile(conf_array, 25)), 4),
        "confidence_p75": round(float(np.percentile(conf_array, 75)), 4),
        "major_field_distribution": dict(Counter(r["major_field"] for r in silver_labels).most_common()),
        "detailed_field_count": len(set(r["detailed_field"] for r in silver_labels)),
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Detailed Silver Label Stats ===")
    print(f"Total abstracts: {stats['total_abstracts']}")
    print(f"Above threshold ({args.threshold}): {stats['above_threshold']} ({stats['yield_pct']}%)")
    print(f"Confidence: mean={stats['confidence_mean']}, median={stats['confidence_median']}")
    print(f"Detailed fields covered: {stats['detailed_field_count']}")
    print(f"\nTop major fields:")
    for field, n in Counter(r["major_field"] for r in silver_labels).most_common(15):
        print(f"  {field}: {n}")


if __name__ == "__main__":
    main()
