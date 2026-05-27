"""
Step 3: Classify TACC abstracts via nearest-sub-field matching.

Loads the FAISS index + metadata, embeds each abstract from
database_abstracts.xlsx, retrieves top-10 nearest CIP entries,
and assigns the majority Major_Field_label as prediction.
"""

import json
from collections import Counter

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# --- Configuration ---
TOP_K = 10
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Auto-detect hardware
if torch.cuda.is_available():
    BATCH_SIZE = 32
    DEVICE = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    BATCH_SIZE = 8
    DEVICE = "cpu"
    print("Using CPU")

# --- Load resources ---

print("Loading FAISS index and metadata...")
index = faiss.read_index("cip_index.faiss")
with open("cip_metadata.json", "r") as f:
    metadata = json.load(f)

print(f"Index: {index.ntotal} vectors")

print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=DEVICE)

# Load valid major fields for validation
with open("major_fields.json", "r") as f:
    valid_major_fields = set(json.load(f))

# --- Load abstracts ---

print("Loading abstracts from database_abstracts.xlsx...")
df = pd.read_excel("database_abstracts.xlsx")
abstracts = df["abstract"].fillna("").tolist()
existing_labels = df["major_field"].tolist()

print(f"Loaded {len(abstracts)} abstracts")

# --- Classify ---

print(f"Classifying abstracts (batch_size={BATCH_SIZE})...")

# Prepend query prefix for BGE model
queries = [QUERY_PREFIX + abstract for abstract in abstracts]

# Embed all abstracts
embeddings = model.encode(
    queries,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=BATCH_SIZE,
)
embeddings = np.array(embeddings, dtype=np.float32)

# Query FAISS for top-K
similarities, indices = index.search(embeddings, TOP_K)

# --- Aggregate results ---

results = []
for i in range(len(abstracts)):
    top_k_indices = indices[i]
    top_k_sims = similarities[i]

    # Collect major fields from top-K
    top_k_fields = [metadata[idx]["Major_Field_label"] for idx in top_k_indices]

    # Majority vote
    field_counts = Counter(top_k_fields)
    predicted_field, majority_count = field_counts.most_common(1)[0]

    results.append({
        "abstract": abstracts[i],
        "existing_label": existing_labels[i] if pd.notna(existing_labels[i]) else None,
        "predicted_field": predicted_field,
        "top1_similarity": float(top_k_sims[0]),
        "agreement_ratio": majority_count / TOP_K,
        "top10_fields": top_k_fields,
        "top1_detailed_field": metadata[top_k_indices[0]]["Detailed_Field_label"],
        "top1_cip_title": metadata[top_k_indices[0]]["SED_CIPTitle"],
    })

# Validate all predictions are in valid set
invalid = [r for r in results if r["predicted_field"] not in valid_major_fields]
if invalid:
    print(f"WARNING: {len(invalid)} predictions not in valid major fields set!")
else:
    print("All predictions are valid major fields.")

# --- Save results ---

output_file = "classification_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} results to {output_file}")

# Summary stats
has_label = [r for r in results if r["existing_label"] is not None]
print(f"  Abstracts with existing labels: {len(has_label)}")
print(f"  Mean top-1 similarity: {np.mean([r['top1_similarity'] for r in results]):.4f}")
print(f"  Mean agreement ratio: {np.mean([r['agreement_ratio'] for r in results]):.2f}")
print("Done.")
