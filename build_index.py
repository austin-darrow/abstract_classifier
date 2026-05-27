"""
Step 1-2: Prepare taxonomy reference set and build FAISS index.

Loads CIP taxonomy from SEDCIP24.json, constructs enriched text strings
combining hierarchy + definition, embeds them with BAAI/bge-large-en-v1.5,
and saves a FAISS index + metadata for fast retrieval.
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Step 1: Load and enrich taxonomy ---

with open("SEDCIP24.json", "r") as f:
    cip_entries = json.load(f)

print(f"Loaded {len(cip_entries)} CIP entries")

# Build enriched text and metadata
enriched_texts = []
metadata = []

for entry in cip_entries:
    text = (
        f"Broad field: {entry['Broad_Field_label']} | "
        f"Major field: {entry['Major_Field_label']} | "
        f"Detailed field: {entry['Detailed_Field_label']} | "
        f"{entry['SED_CIPTitle']}: {entry['CIPDefinition']}"
    )
    enriched_texts.append(text)
    metadata.append({
        "Major_Field_label": entry["Major_Field_label"],
        "Detailed_Field_label": entry["Detailed_Field_label"],
        "SED_CIPTitle": entry["SED_CIPTitle"],
    })

print(f"Built {len(enriched_texts)} enriched text strings")
print(f"\nSample enriched text (first entry):\n{enriched_texts[0][:200]}...")

# --- Step 2: Embed and build FAISS index ---

print("\nLoading embedding model (BAAI/bge-large-en-v1.5)...")
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

print("Encoding taxonomy entries...")
embeddings = model.encode(
    enriched_texts,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=64,
)

embeddings = np.array(embeddings, dtype=np.float32)
print(f"Embeddings shape: {embeddings.shape}")

# Build FAISS index (inner product on normalized vectors = cosine similarity)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print(f"FAISS index built: {index.ntotal} vectors, dimension {dimension}")

# Save index and metadata
faiss.write_index(index, "cip_index.faiss")
with open("cip_metadata.json", "w") as f:
    json.dump(metadata, f)

print("\nSaved: cip_index.faiss, cip_metadata.json")
print("Done.")
