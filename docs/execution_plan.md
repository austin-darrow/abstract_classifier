# CIP Classifier — Execution Plan

## Objective

Classify research abstracts into CIP taxonomy fields (74 major fields, 22 broad fields). Train on LLM-generated synthetic abstracts + high-confidence real TACC pseudo-labels, evaluate on manually-verified subset of real TACC abstracts.

## Success Criteria

| Metric | Target | Current Best | Notes |
|--------|--------|--------------|-------|
| Major field accuracy (verified subset) | >70% | 34.3% (B0) | Evaluated on hand-labeled gold set |
| Broad field accuracy (verified subset) | >90% | 59.8% (B0) | Same gold set |
| Inter-model agreement rate | >60% | TBD | B4/B5 agree on real TACC |
| Silver label set size | >5,000 | TBD | High-confidence agreed predictions |

**Key insight:** The classifier is likely more accurate than existing TACC database labels. DB labels are NOT treated as ground truth.

---

## Data

| Dataset | Size | Purpose |
|---------|------|---------|
| Synthetic train | ~16,183 | Base training data |
| Synthetic test | ~4,054 | Development evaluation |
| Real TACC abstracts | 16,209 (46 of 74 fields covered) | Silver label source + final evaluation |
| Silver labels | TBD (target >5,000) | High-confidence pseudo-labels for retraining |
| Gold set | TBD (~200-500) | Hand-verified labels for true evaluation |

- Generation: DeepSeek-R1 (671B, FP8) on Vista GH200 × 9 nodes
- Adversarial verification: same model verifies each abstract against sibling fields
- Train/test split: 80/20 stratified by major field
- Silver labels: model consensus (B4 + B5 agree) at confidence > threshold

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

## Phase B: Classifier Approaches ✅ DONE (archived)

All approaches evaluated. B4 (SetFit) and B5 (SciBERT fine-tune) selected as most promising. Others archived.

| # | Approach | Synth Major Acc | Real TACC Major Acc | Status |
|---|----------|-----------------|---------------------|--------|
| B0 | FAISS baseline (CIP defs) | 52.2% | **34.3%** | Archived (reference) |
| B1 | kNN on synthetic | 85.6% | 25.9% | Archived |
| B2 | TF-IDF + LogReg | 82.2% | 24.9% | Archived |
| B3 | Embedding head (MLP) | 88.6% | 19.6% | Archived |
| B4 | **SetFit** | 91.9% | 25.8% | **Active** |
| B5 | **SciBERT fine-tune** | 90.8% | **32.3%** | **Active** |
| B6 | Zero-shot LLM (32B) | 15.7% | 31.3% | Archived |

**Key findings:**
- Inverse correlation: more synthetic accuracy → worse real accuracy (distribution shift)
- B0 baseline beats all trained models on real data — problem is training distribution, not model capacity
- Same fields get 0% F1 across ALL approaches (ME, CS, Bio general) — likely mislabeled in DB
- Classifier appears more accurate than database labels based on manual review

---

## Phase C: Close the Synthetic→Real Gap

### C0. Repo Cleanup ✅ DONE

| Step | Action | Status |
|------|--------|--------|
| C0.1 | Archive B2 (tfidf), B3 (embedding_head), B6 (zeroshot) code → `archive/` | ✅ |
| C0.2 | Archive all prediction/metric files for B0-B3, B6 → `archive/output/predictions/` | ✅ |
| C0.3 | Archive completed generation SLURM scripts → `archive/slurm/` | ✅ |
| C0.4 | Update model dispatcher to remove embedding_head, add finetune | ✅ |

---

### C1. SetFit Upgrade (bge-large) ✅ CONFIG DONE

Fix apples-to-oranges: SetFit previously used `bge-base-en-v1.5` while all other approaches used `bge-large-en-v1.5`.

| Step | Action | Status |
|------|--------|--------|
| C1.1 | Update `configs/train.yaml`: encoder → `BAAI/bge-large-en-v1.5` | ✅ |
| C1.2 | Re-run SetFit training on synthetic data | ⬜ Run on Vista |
| C1.3 | Evaluate on synthetic test + real TACC | ⬜ |
| C1.4 | Compare to previous bge-base results | ⬜ |

**Expected:** ~3 hr training on GH200. Larger embeddings (1024-dim vs 768-dim) should improve separation.

---

### C2. Label Quality Audit

**Goal:** Quantify label noise in TACC data. Determine which 0% F1 fields are genuinely hard vs. mislabeled.

| Step | Action | Status |
|------|--------|--------|
| C2.1 | Run `scripts/audit_labels.py` on B4/B5 predictions | ⬜ |
| C2.2 | Review diagnostic charts (confidence vs. agreement, per-field plots) | ⬜ |
| C2.3 | For problem fields (ME, CS, Bio general): read samples, determine if DB or model is right | ⬜ |
| C2.4 | Establish per-field "label trust score" | ⬜ |
| C2.5 | Determine confidence threshold for silver labels (informed by audit) | ⬜ |

**Charts produced:**
- `confidence_vs_db_agreement.png` — At each threshold, what % of agreed predictions match DB?
- `agreement_rate_by_field.png` — Which fields do models agree on most?
- `problem_field_*.png` — What do models predict instead for 0% F1 fields?
- `confidence_distribution_agreement.png` — Confidence histogram split by DB match

**Script:** `scripts/audit_labels.py`

---

### C3. Silver Labels & Semi-Supervised Retraining

**Goal:** Bridge synthetic→real gap by training on real TACC abstracts with pseudo-labels.

**Depends on:** C1 (upgraded SetFit), C2 (validated threshold).

| Step | Action | Status |
|------|--------|--------|
| C3.1 | Run `scripts/build_silver_labels.py --threshold <from C2>` | ⬜ |
| C3.2 | Verify silver label quality (spot check, field distribution) | ⬜ |
| C3.3 | Create combined training set: synthetic + silver | ⬜ |
| C3.4 | Retrain B5 (SciBERT) on combined data | ⬜ |
| C3.5 | Retrain B4 (SetFit) on combined data | ⬜ |
| C3.6 | Evaluate on held-out real TACC (exclude silver records) | ⬜ |
| C3.7 | Self-training round 2 (optional): retrained model → new silver labels → retrain | ⬜ |

**Silver label criteria:** B4 and B5 agree on major field AND max(confidence) > threshold.

**Script:** `scripts/build_silver_labels.py`

---

### C4. Hyperparameter Tuning

**Goal:** Squeeze more from B4/B5 after silver labels are incorporated.

**Depends on:** C3 (retrained models as new baseline).

| Param | B4 SetFit | B5 SciBERT |
|-------|-----------|------------|
| Current | 20 iter, 1 epoch | 3 epochs, lr=2e-5 |
| Try | 40/60 iter, 2-3 epochs | 5-8 epochs, lr={1e-5, 5e-5} |
| New | Hard-negative mining (confusing pairs) | label_smoothing=0.1, DeBERTa-v3-base |

**Charts needed:**
- Learning curves (acc vs. training set size at 25/50/75/100%)
- Per-field F1 bar chart (B4 vs B5 side-by-side)
- Top-20 confusion pairs for each model

---

### C5. Hierarchical Classification

**Goal:** Exploit the consistent 15-20pt broad→major accuracy gap. Two-stage classification.

**Depends on:** C3/C4 (need best flat model as comparison baseline).

| Step | Action | Status |
|------|--------|--------|
| C5.1 | Train broad-field classifier (22 classes) — SciBERT on synthetic + silver | ⬜ |
| C5.2 | Train per-broad-field major classifiers (2-5 classes each) | ⬜ |
| C5.3 | Inference pipeline: broad → route → major | ⬜ |
| C5.4 | End-to-end evaluation vs. flat models | ⬜ |
| C5.5 | Error analysis: does hierarchical reduce cross-broad confusion? | ⬜ |

**Architecture:**
- Stage 1: SciBERT fine-tune (22 broad classes)
- Stage 2: Per-broad SciBERT or SetFit (2-5 major classes each)
- Broad fields with only 1 major field: skip Stage 2, pass through

**Module:** `src/cip_classifier/models/hierarchical.py`

---

## Execution Flow

```
C0 (cleanup) ✅
│
├── C1 (SetFit bge-large) ──┐
│                            ├──► C3 (silver labels + retrain)
└── C2 (label audit) ───────┘          │
                                        ├──► C4 (hyperparam tuning)
                                        │          │
                                        └──────────┴──► C5 (hierarchical)
```

---

## Infrastructure

- **Training:** Vista single GH200 node (120GB GPU memory)
- **Active code:** `baselines/finetune.py` (B5), `baselines/setfit_classify.py` (B4), all evaluation
- **Archived:** `archive/` — retired approaches (B2, B3, B6), old predictions, generation scripts

---

## Key Decisions

1. **DB labels are not ground truth** — evaluate against manual audits + inter-model consensus
2. **Silver labels via model consensus** — B4+B5 agree at high confidence, not single-model confidence alone
3. **Focus on B4 (SetFit) + B5 (SciBERT) only** — other approaches archived
4. **No calibration against DB labels** — poisoned by label noise; use inter-model agreement instead
5. **Hierarchical is last** — validate flat-model improvements first (C3/C4), then try hierarchical (C5)
6. **Self-training capped at 2-3 rounds** — diminishing returns, risk of error propagation
7. **UNASSIGNED records (n=419) excluded** from all evaluation going forward
8. **Stop when good enough** — no over-engineering past success criteria
