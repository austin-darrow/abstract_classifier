# CIP Classifier — Execution Plan

## Objective

Classify research abstracts into CIP taxonomy fields (74 major fields, 22 broad fields). Train on LLM-generated synthetic abstracts, evaluate on real TACC abstracts.

## Success Criteria

| Metric | Target | Current Baseline |
|--------|--------|-----------------|
| Major field accuracy (real TACC) | >70% | 29% |
| Broad field accuracy (real TACC) | >90% | ~50% |

**Stop progressing through approaches once criteria are met.**

---

## Data

| Dataset | Size | Purpose |
|---------|------|---------|
| Synthetic train | ~18,800 (at 10/CIP) | Training |
| Synthetic test | ~4,700 (at 10/CIP) | Development evaluation |
| Real TACC abstracts | 16,209 (46 of 74 fields covered) | True evaluation target |

- Generation: DeepSeek-R1 (671B, FP8) on Vista GH200 × 9 nodes
- Adversarial verification: same model verifies each abstract against sibling fields
- Train/test split: 80/20 stratified by major field
- Class imbalance handled at training time (class weights / oversampling)

---

## Phase A: Evaluation Framework ✅ DONE

Unified framework so every approach produces comparable results automatically.

| Step | Description | Status |
|------|-------------|--------|
| A1 | Standardized prediction format (`Prediction`, `PredictionSet`) | ✅ |
| A2 | Unified metrics (accuracy, F1, top-k, per-field, confusion) | ✅ |
| A3 | Multi-model comparison tables + agreement analysis | ✅ |
| A4 | Visualization (bar charts, confusion matrices, histograms) | ✅ |
| A5 | Training diagnostics (learning curves, error analysis) | ✅ |
| A6 | CLI: `cip-classifier compare` | ✅ |

---

## Phase B: Classifier Approaches (easiest → hardest)

Each approach outputs a `PredictionSet`. After each, run comparison and check targets.

### B1. kNN on Synthetic Abstracts

- Embed training abstracts with SentenceTransformer → FAISS index
- Classify by majority vote of top-k nearest neighbors
- Tests: Do domain embeddings + more data beat CIP definition matching?
- Effort: ~10 min
- Implementation: `baselines/faiss_retrieval.py` → `build_index_from_abstracts()`

### B2. TF-IDF + Logistic Regression

- `TfidfVectorizer(max_features=50000, ngram_range=(1,2))` + `LogisticRegression`
- Classical ML baseline — trains in seconds
- Tests: Can bag-of-words features solve this?
- Effort: ~10 min
- Implementation: `baselines/tfidf_logreg.py`

### B3. Embedding Head (Frozen Encoder + MLP)

- Encode all abstracts once with frozen SentenceTransformer
- Train 2-layer MLP classification head (hidden_dim → n_classes)
- Tests: Does a learned decision boundary beat kNN?
- Effort: ~30 min
- Implementation: `models/embedding_head.py`

### B4. SetFit (Contrastive Fine-Tuning)

- Fine-tune encoder via contrastive learning on training pairs
- Train classification head on adapted embeddings
- Tests: Does adapting the encoder improve over frozen?
- Effort: ~30 min
- Implementation: `models/setfit_model.py`

### B5. Full Fine-Tune (SciBERT / DeBERTa)

- Unfreeze entire encoder + classification head, train end-to-end
- HuggingFace Trainer with warmup, cosine decay, early stopping
- Candidates: `allenai/scibert_scivocab_uncased`, `microsoft/deberta-v3-base`
- Tests: Best single-model performance
- Effort: ~1-2 hr
- Implementation: `models/finetune.py`

### B6. Zero-Shot LLM (Accuracy Ceiling)

- Prompt a 7B-14B model with taxonomy + abstract → parse field name
- Not for production — establishes theoretical maximum
- Tests: What's the best possible accuracy with full taxonomy context?
- Effort: ~30 min
- Implementation: `baselines/llm_zeroshot.py`

---

## Phase C: Iterate (only if Phase B doesn't meet targets)

| Step | Trigger | Action |
|------|---------|--------|
| C1 | Best approach accuracy still climbing at 100% data | Learning curve analysis → generate more data |
| C2 | Specific fields have low accuracy | Targeted augmentation for those fields |
| C3 | Most errors cross broad-field boundaries | Hierarchical: broad field first, then major within |
| C4 | Approaches have complementary errors | Ensemble (majority vote or stacking) |

---

## Execution Flow

```
Data Generation (Vista, ~10-14 hrs remaining)
│
├─ Phase A: Evaluation Framework ✅ DONE
│
└─ Phase B: Classifiers (after data ready)
   │
   B1 (kNN synth) ──► compare ──► targets met? ──► STOP
   │                                    │ no
   B2 (TF-IDF) ────► compare ──► targets met? ──► STOP
   │                                    │ no
   B3 (emb head) ──► compare ──► targets met? ──► STOP
   │                                    │ no
   B4 (SetFit) ────► compare ──► targets met? ──► STOP
   │                                    │ no
   B5 (fine-tune) ─► compare ──► targets met? ──► STOP
   │                                    │ no
   B6 (LLM ceil.) ─► compare ──► Phase C
```

---

## Infrastructure

- **Generation:** Vista GH200, 9 nodes, DeepSeek-R1 671B FP8, vLLM pipeline-parallel
- **Training:** Vista single GH200 node (120GB GPU memory)
- **Concurrency:** 48 async requests during generation
- **Resume logic:** Pipeline skips already-generated abstracts automatically

---

## Key Decisions

1. **Keep all approaches on the table** — progress easiest→hardest, let metrics decide
2. **Stop when good enough** — no over-engineering past success criteria
3. **Evaluation framework first** — every approach uses same format, comparisons are automatic
4. **10/CIP uniformly** — ensures minimum viable representation for all 74 fields
5. **Class imbalance at training time** — class weights/oversampling, not more generation for thin fields
6. **Real TACC abstracts are the true metric** — synthetic test set is for development only
7. **Learning curves drive data decisions** — don't guess, measure
