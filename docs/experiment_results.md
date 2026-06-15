# CIP Classifier — Experiment Results

## Summary Table

| # | Approach | Synthetic Test ||| Real TACC |||
|---|----------|---|---|---|---|---|---|
| | | Major Acc | Broad Acc | Macro F1 | Major Acc | Broad Acc | Macro F1 |
|---|----------|-----------|-----------|----------|-----------|-----------|----------|
| B0 | FAISS baseline (CIP defs) | — | — | — | 0.29 | ~0.50 | — |
| B1 | kNN on synthetic abstracts | | | | | | |
| B2 | TF-IDF + LogReg | | | | | | |
| B3 | Embedding head (frozen + MLP) | | | | | | |
| B4 | SetFit | | | | | | |
| B5 | SciBERT fine-tune | | | | | | |
| B6 | Zero-shot LLM (ceiling) | | | | | | |

**Targets:** Major ≥ 0.70, Broad ≥ 0.90 on real TACC abstracts.

---

## Detailed Results

### B0. FAISS Baseline (CIP Definitions)

- **Date:** 2025 (prior work)
- **Encoder:** `BAAI/bge-large-en-v1.5`
- **Method:** Embed CIP taxonomy entries → FAISS index, embed abstracts → top-10 nearest neighbors → majority vote on major field
- **Training data:** None (zero-shot retrieval from taxonomy)

| Dataset | Major Acc | Broad Acc | Macro F1 | Top-3 Acc | Top-5 Acc |
|---------|-----------|-----------|----------|-----------|-----------|
| Real TACC | 0.29 | ~0.50 | — | — | — |

**Notes:**
- Matching abstracts to definitions is inherently hard — definitions are terse
- Serves as the "before" measurement

---

### B1. kNN on Synthetic Abstracts

- **Date:**
- **Encoder:**
- **Method:** Embed synthetic training abstracts → FAISS index, classify via top-k majority vote
- **Training data:** ~18,800 synthetic abstracts (10/CIP)
- **Hyperparams:** top_k=

| Dataset | Major Acc | Broad Acc | Macro F1 | Top-3 Acc | Top-5 Acc |
|---------|-----------|-----------|----------|-----------|-----------|
| Synthetic test | | | | | |
| Real TACC | | | | | |

**Notes:**


**Decision:** ☐ Meets targets → STOP | ☐ Continue to B2

---

### B2. TF-IDF + Logistic Regression

- **Date:**
- **Method:** TF-IDF vectorization + LogisticRegression
- **Training data:** ~18,800 synthetic abstracts
- **Hyperparams:** max_features=50000, ngram_range=(1,2), max_iter=1000, class_weight=balanced

| Dataset | Major Acc | Broad Acc | Macro F1 | Top-3 Acc | Top-5 Acc |
|---------|-----------|-----------|----------|-----------|-----------|
| Synthetic test | | | | | |
| Real TACC | | | | | |

**Notes:**


**Decision:** ☐ Meets targets → STOP | ☐ Continue to B3

---

### B3. Embedding Head (Frozen Encoder + MLP)

- **Date:**
- **Encoder:**
- **Method:** Frozen SentenceTransformer embeddings → 2-layer MLP head
- **Training data:** ~18,800 synthetic abstracts
- **Hyperparams:** hidden_dim=, dropout=, lr=, epochs=, batch_size=

| Dataset | Major Acc | Broad Acc | Macro F1 | Top-3 Acc | Top-5 Acc |
|---------|-----------|-----------|----------|-----------|-----------|
| Synthetic test | | | | | |
| Real TACC | | | | | |

**Notes:**


**Decision:** ☐ Meets targets → STOP | ☐ Continue to B4

---

### B4. SetFit (Contrastive Fine-Tuning)

- **Date:**
- **Encoder:**
- **Method:** Contrastive fine-tune on training pairs → classification head on adapted embeddings
- **Training data:** ~18,800 synthetic abstracts
- **Hyperparams:** num_iterations=, num_epochs=

| Dataset | Major Acc | Broad Acc | Macro F1 | Top-3 Acc | Top-5 Acc |
|---------|-----------|-----------|----------|-----------|-----------|
| Synthetic test | | | | | |
| Real TACC | | | | | |

**Notes:**


**Decision:** ☐ Meets targets → STOP | ☐ Continue to B5

---

### B5. Full Fine-Tune (SciBERT / DeBERTa)

- **Date:**
- **Model:**
- **Method:** Full encoder fine-tune + classification head via HuggingFace Trainer
- **Training data:** ~18,800 synthetic abstracts
- **Hyperparams:** lr=, warmup_ratio=, weight_decay=, max_length=, epochs=, batch_size=, early_stopping_patience=

| Dataset | Major Acc | Broad Acc | Macro F1 | Top-3 Acc | Top-5 Acc |
|---------|-----------|-----------|----------|-----------|-----------|
| Synthetic test | | | | | |
| Real TACC | | | | | |

**Notes:**


**Decision:** ☐ Meets targets → STOP | ☐ Continue to B6

---

### B6. Zero-Shot LLM (Accuracy Ceiling)

- **Date:**
- **Model:**
- **Method:** Prompt with full taxonomy + abstract → parse predicted field name
- **Training data:** None (zero-shot)
- **Hyperparams:** temperature=, max_tokens=

| Dataset | Major Acc | Broad Acc | Macro F1 | Top-3 Acc | Top-5 Acc |
|---------|-----------|-----------|----------|-----------|-----------|
| Synthetic test | | | | | |
| Real TACC | | | | | |

**Notes:**


**Decision:** ☐ Meets targets → STOP | ☐ Proceed to Phase C

---

## Phase C Experiments (if needed)

### C1. Learning Curve

| Fraction | n_train | Major Acc | Broad Acc | Macro F1 |
|----------|---------|-----------|-----------|----------|
| 0.10 | | | | |
| 0.25 | | | | |
| 0.50 | | | | |
| 0.75 | | | | |
| 1.00 | | | | |

**Conclusion:** ☐ Saturated (more data won't help) | ☐ Still climbing (generate more)

---

### C2. Targeted Augmentation

| Field | Before Acc | After Acc | Samples Added |
|-------|-----------|-----------|---------------|
| | | | |

---

### C3. Hierarchical Classification

| Level | Accuracy | Notes |
|-------|----------|-------|
| Broad field (22 classes) | | |
| Major within broad | | |
| Combined pipeline | | |

---

### C4. Ensemble

| Method | Major Acc | Broad Acc | Components |
|--------|-----------|-----------|------------|
| Majority vote | | | |
| Stacking | | | |

---

## Error Analysis Notes

### Cross-Broad-Field Errors
- Total errors:
- Within same broad field: ( %)
- Across broad fields: ( %)

### Hardest Fields (lowest F1)

| Field | F1 | Support | Top Confusion |
|-------|-----|---------|---------------|
| | | | |

### Common Confusion Pairs

| True Field | Predicted Field | Count |
|-----------|----------------|-------|
| | | |
