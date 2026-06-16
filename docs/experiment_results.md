# CIP Classifier — Experiment Results

## Summary Table

| # | Approach | Synthetic Test ||| Real TACC |||
|---|----------|---|---|---|---|---|---|
| | | Major Acc | Broad Acc | Macro F1 | Major Acc | Broad Acc | Macro F1 |
|---|----------|-----------|-----------|----------|-----------|-----------|----------|
| B0 | FAISS baseline (CIP defs) | 0.5224 | 0.5560 | 0.4363 | 0.3431 | 0.5981 | 0.1713 |
| B1 | kNN on synthetic abstracts | 0.8564 | 0.8661 | 0.8500 | 0.2592 | 0.4265 | 0.1580 |
| B2 | TF-IDF + LogReg | 0.8219 | 0.8355 | 0.8403 | 0.2485 | 0.4212 | 0.1353 |
| B3 | Embedding head (frozen + MLP) | 0.8858 | 0.8959 | 0.8701 | 0.1962 | 0.3644 | 0.1111 |
| B4 | SetFit | 0.9188 | 0.9265 | 0.8448 | 0.2579 | 0.4424 | 0.1609 |
| B5 | SciBERT fine-tune | 0.9077 | 0.9171 | 0.8428 | 0.3228 | 0.5295 | 0.1775 |
| B6 | Zero-shot LLM (ceiling) | 0.1570 | 0.2240 | 0.3062 | 0.3130 | 0.5750 | 0.2534 |

**Targets:** Major ≥ 0.70, Broad ≥ 0.90 on real TACC abstracts.

---

## Detailed Results

### B0. FAISS Baseline (CIP Definitions)

- **Date:** 2026-06-15
- **Encoder:** `BAAI/bge-large-en-v1.5`
- **Method:** Embed CIP taxonomy entries (2,355 vectors) → FAISS index, embed abstracts → top-10 nearest neighbors → majority vote on major field
- **Training data:** None (zero-shot retrieval from taxonomy definitions)

| Dataset | Major Acc | Broad Acc | Macro F1 | Top-3 Acc | Top-5 Acc |
|---------|-----------|-----------|----------|-----------|----------|
| Synthetic test (n=4054) | 0.5224 | 0.5560 | 0.4363 | 0.8592 | 0.9262 |
| Real TACC (n=16209) | 0.3431 | 0.5981 | 0.1713 | 0.5351 | 0.6069 |

**Notes:**
- **B0 is the best on real TACC so far** — 34.3% major, 59.8% broad beats all synthetic-trained models
- Synthetic test is only 52% — CIP definitions don't match synthetic abstract language well
- But real TACC has 60% broad accuracy — taxonomy definitions are actually more aligned with real abstracts
- Top-3 real TACC at 53.5%, Top-5 at 60.7% — correct field is often in the neighborhood
- Confusion is between semantically close fields: Materials↔Physics, Chemistry↔Biochem, CS↔CS-other
- Mechanical Engineering (n=933) gets 0% — consistently the hardest field across all approaches
- Zero-shot with no training data, yet beats B1-B3 on real data — confirms synthetic-trained models overfit
- This reframes the problem: **learn to improve B0, not replace it**

---

### B1. kNN on Synthetic Abstracts

- **Date:** 2026-06-15
- **Encoder:** `BAAI/bge-large-en-v1.5`
- **Method:** Embed synthetic training abstracts → FAISS index, classify via top-k majority vote
- **Training data:** 16,183 synthetic abstracts (10/CIP)
- **Hyperparams:** top_k=10

| Dataset | Major Acc | Broad Acc | Macro F1 | Top-3 Acc | Top-5 Acc |
|---------|-----------|-----------|----------|-----------|-----------|
| Synthetic test (n=4054) | 0.8564 | 0.8661 | 0.8500 | 0.9896 | 0.9914 |
| Real TACC (n=16209) | 0.2592 | 0.4265 | 0.1580 | 0.4058 | 0.4446 |

**Notes:**
- Synthetic test accuracy is strong (86%) — embeddings separate well within synthetic domain
- Real TACC major accuracy (26%) is WORSE than B0 baseline (29%) — significant domain gap
- Real TACC broad accuracy (43%) also below B0 (~50%)
- Top confused pairs on real data: CS/Physics/Materials → "Technology and technical fields" and "Science-related technologies"
- The kNN is biased toward synthetic-heavy fields (Technology, Health, Clinical medical research)
- Top-5 on real TACC is 44% — correct field is often in the neighborhood but not majority
- "Biological and biomedical sciences, general" (n=566) gets 0% — entirely eaten by more specific fields

**Decision:** ☐ Meets targets → STOP | ☒ Continue to B2

---

### B2. TF-IDF + Logistic Regression

- **Date:** 2026-06-15
- **Method:** TF-IDF vectorization + LogisticRegression (class_weight=balanced, solver=lbfgs)
- **Training data:** 16,183 synthetic abstracts
- **Hyperparams:** max_features=50000, ngram_range=(1,2), max_iter=1000, sublinear_tf=True

| Dataset | Major Acc | Broad Acc | Macro F1 | Top-3 Acc | Top-5 Acc |
|---------|-----------|-----------|----------|-----------|----------|
| Synthetic test (n=4054) | 0.8219 | 0.8355 | 0.8403 | 0.9763 | 0.9919 |
| Real TACC (n=16209) | 0.2485 | 0.4212 | 0.1353 | 0.3682 | 0.4356 |

**Notes:**
- Synthetic test slightly lower than B1 kNN (82% vs 86% major) — kNN benefits from exact embedding matches
- Real TACC (25% major) comparable to B1 (26%) — both hampered by same domain gap
- Massive "Health sciences, other" attractor on real data — CS, biochem, neuro, physics all flowing into it
- Many fields get 0% F1 on real data (Bioinformatics n=559, Education n=53, etc.)
- Confusion pattern differs from B1: B2 has Health Sciences gravity well; B1 had Technology bias
- Top-5 real TACC (44%) identical to B1 — confirms ceiling for approaches trained only on synthetic data
- Training: ~5 seconds end-to-end (no GPU needed)

**Decision:** ☐ Meets targets → STOP | ☒ Continue to B3

---

### B3. Embedding Head (Frozen Encoder + MLP)

- **Date:** 2026-06-15
- **Encoder:** `BAAI/bge-large-en-v1.5` (frozen)
- **Method:** Frozen SentenceTransformer embeddings → 2-layer MLP head (256→256→74)
- **Training data:** 16,183 synthetic abstracts
- **Hyperparams:** hidden_dim=256, dropout=0.1, lr=0.001, epochs=20, batch_size=64

| Dataset | Major Acc | Broad Acc | Macro F1 | Top-3 Acc | Top-5 Acc |
|---------|-----------|-----------|----------|-----------|----------|
| Synthetic test (n=4054) | 0.8858 | 0.8959 | 0.8701 | 0.9933 | 0.9990 |
| Real TACC (n=16209) | 0.1962 | 0.3644 | 0.1111 | 0.3431 | 0.4279 |

**Notes:**
- Best synthetic test score so far (88.6% major) — MLP learns better boundaries than kNN/TF-IDF within-domain
- Real TACC is WORST so far (19.6% major, 36.4% broad) — **overfits harder to synthetic distribution**
- MLP memorizes synthetic patterns that don't transfer: Technology attractor pulls CS/ME, Microbiology pulls Biochem
- "Biological and biomedical sciences, general" (n=566) again 0% — a consistent failure across all approaches
- Training converged to 90% train acc in 20 epochs — not underfitting, just domain-shifted
- The non-linear decision boundary hurts generalization compared to simpler B1/B2 on OOD data
- Confirms: more expressive model on synthetic-only data = more overfitting to synthetic quirks

**Decision:** ☐ Meets targets → STOP | ☒ Continue to B4

---

### B4. SetFit (Contrastive Fine-Tuning)

- **Date:** 2026-06-16
- **Encoder:** `BAAI/bge-base-en-v1.5`
- **Method:** Contrastive fine-tune on 647,320 training pairs → LogReg head on adapted embeddings
- **Training data:** 16,183 synthetic abstracts
- **Hyperparams:** num_iterations=20, num_epochs=1, batch_size=16

| Dataset | Major Acc | Broad Acc | Macro F1 | Top-3 Acc | Top-5 Acc |
|---------|-----------|-----------|----------|-----------|----------|
| Synthetic test (n=4054) | 0.9188 | 0.9265 | 0.8448 | 0.9768 | 0.9845 |
| Real TACC (n=16209) | 0.2579 | 0.4424 | 0.1609 | 0.4253 | 0.5156 |

**Notes:**
- Best synthetic accuracy tied with B5 (91.9% vs 90.8%) — contrastive training reshapes embedding space well
- Real TACC (25.8% major, 44.2% broad) — mid-pack, well below B0 (34.3% / 59.8%)
- Training took ~2.5 hours on GH200 (647K contrastive pairs at 4.2 it/s)
- Novel confusion patterns: CS→Interdisciplinary CS, Chemistry→Chemical Engineering, Materials→Science-related Tech
- SetFit adapts embeddings but still learns synthetic-specific boundaries
- Computer Science (n=860) gets 0% F1 — completely absorbed by related CS fields
- 419 UNASSIGNED records all get 0% — expected, no training signal for this

**Decision:** ☐ Meets targets → STOP | ☒ Continue to B5

---

### B5. Full Fine-Tune (SciBERT / DeBERTa)

- **Date:** 2026-06-15
- **Model:** `allenai/scibert_scivocab_uncased`
- **Method:** Full encoder fine-tune + classification head via HuggingFace Trainer
- **Training data:** 16,183 synthetic abstracts (14,564 train / 1,619 val)
- **Hyperparams:** lr=2e-5, warmup_ratio=0.1, weight_decay=0.01, max_length=512, epochs=3, batch_size=16, early_stopping_patience=3

| Dataset | Major Acc | Broad Acc | Macro F1 | Top-3 Acc | Top-5 Acc |
|---------|-----------|-----------|----------|-----------|----------|
| Synthetic test (n=4054) | 0.9077 | 0.9171 | 0.8428 | 0.9899 | 0.9951 |
| Real TACC (n=16209) | 0.3228 | 0.5295 | 0.1775 | 0.4899 | 0.5799 |

**Notes:**
- Best synthetic accuracy (90.8%) — full fine-tune clearly separates synthetic classes best
- Real TACC (32.3% major, 53.0% broad) still below B0 baseline (34.3% / 59.8%)
- Trained in only 108 seconds on GH200 — extremely fast
- Epoch progression: val_acc 82.4% → 89.0% → 90.6% (no overfitting within 3 epochs)
- Same Mechanical Engineering (n=933) at 0% F1 pattern — structurally hard field
- Confusion is semantically reasonable: ME→Physics, Materials→Physics, CS→Technology
- Full fine-tune adapts the encoder but still can't bridge the synthetic→real gap
- **B0 remains the real-TACC leader** — the problem isn't model capacity, it's training distribution

**Decision:** ☐ Meets targets → STOP | ☒ Continue to B6

---

### B6. Zero-Shot LLM (Accuracy Ceiling)

- **Date:** 2026-06-15
- **Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
- **Method:** Prompt with full taxonomy + abstract → parse predicted field name from JSON response
- **Training data:** None (zero-shot)
- **Hyperparams:** temperature=0.0, max_tokens=2048, concurrency=16, n=1000 subsample

| Dataset | Major Acc | Broad Acc | Macro F1 | Top-3 Acc | Top-5 Acc |
|---------|-----------|-----------|----------|-----------|----------|
| Synthetic test (n=1000) | 0.1570 | 0.2240 | 0.3062 | 0.1570 | 0.1570 |
| Real TACC (n=1000) | 0.3130 | 0.5750 | 0.2534 | 0.3130 | 0.3130 |

**Notes:**
- Parse fix worked: 0 errors on synthetic, 6/1000 on real
- Real TACC (31.3% major, 57.5% broad) — competitive with B0 (34.3%/59.8%) but doesn't beat it
- Synthetic is terrible (15.7%) — model predicts wrong taxonomy names ("Computer and information sciences" vs exact taxonomy labels)
- Massive CS attractor: everything gets classified into CS/CompSci variants
- Confusions are between genuinely close fields: CS↔CS-other, Chemistry↔Materials, Physics↔Materials
- Top-3/5 = Top-1 since LLM only produces one prediction per call (no probabilities)
- The 32B distill is NOT a good ceiling — would need full R1 (671B) or GPT-4 for true ceiling
- Even so, simple FAISS retrieval (B0) matches or beats a 32B reasoning model
- Cost: ~1000 API calls × 2K tokens = expensive for marginal accuracy

**Decision:** ☒ None meet targets → Proceed to Phase C (improve B0)

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
