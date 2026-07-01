# CIP Classifier — Project Report

## 1. Overview

### Problem

TACC receives thousands of HPC allocation requests, each accompanied by a research abstract. These abstracts need to be classified into fields of science using the [CIP taxonomy](https://nces.ed.gov/ipeds/cipcode/) — a three-level hierarchy of **22 broad fields → 74 major fields → 315 detailed fields** — for institutional reporting and resource planning.

Existing database labels for TACC abstracts have a ~55% noise rate (see §3.2), making them unreliable as ground truth. The goal is to build a classifier that's more accurate than the current labels, trained without access to reliably labeled real data.

### Approach

1. **Generate** synthetic training abstracts using DeepSeek-R1 (671B) with adversarial verification
2. **Train** SciBERT-based classifiers on synthetic + pseudo-labeled real data
3. **Evaluate** on a held-out synthetic test set and via inter-model consensus on real TACC data

### Key Results

| Model | Detailed Acc | Major Acc | Macro F1 |
|-------|-------------|-----------|----------|
| Single unified model (D2c, Strategy C) | 87.3% | 93.0% | 0.926 |
| Hierarchical two-model (D2b) | — | 94.1% | 0.936 |
| C4 sweep best (pre-D2 data) | — | 94.0% | 0.937 |

All metrics on synthetic test (n=4,054). The **single unified model** was selected for deployment — 1% behind hierarchical but only requires one model to serve.

### How the Classifier Works

This section explains the deployed model end-to-end, from input text to final prediction.

#### What is SciBERT?

[SciBERT](https://github.com/allenai/scibert) is a pretrained language model built on the BERT architecture. Like BERT, it reads text and produces dense vector representations that capture semantic meaning. What makes SciBERT special is that it was pretrained on 1.14 million scientific papers from Semantic Scholar (covering computer science and biomedical domains), so it already "understands" scientific language before we ever train it on our task.

Under the hood, SciBERT is a 12-layer transformer encoder with ~110M parameters. It processes text as sequences of tokens (subword pieces from a vocabulary of 30,522 scientific terms) and outputs a 768-dimensional vector for each token. The special `[CLS]` token's vector at the end represents the "meaning" of the entire input and is what we use for classification.

#### Architecture

Our model takes SciBERT and adds a **classification head** on top — a single linear layer that maps the 768-dimensional `[CLS]` vector to **315 output logits** (one for each detailed CIP field in the taxonomy).

```
Input abstract (raw text)
    │
    ▼
┌─────────────────────────────────────────┐
│  Tokenizer                              │
│  Split text into subword tokens         │
│  (max 512 tokens)                       │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  SciBERT Encoder (12 transformer layers)│
│                                         │
│  Layers 0–7: FROZEN (not updated)       │
│  Layers 8–11: FINE-TUNED on our data    │
│                                         │
│  Output: 768-dim vector for [CLS] token │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Classification Head                    │
│  Linear: 768 → 315                      │
│  (one logit per detailed CIP field)     │
└─────────────────────────────────────────┘
    │
    ▼
315 raw logits
```

**Why freeze layers 0–7?** The early layers of SciBERT capture general language features (grammar, word meaning) that are already well-learned from pretraining. Freezing them prevents overfitting on our relatively small training set (~23K abstracts). Only the top 4 layers + classification head are updated during training, which acts as a regularizer and was empirically the best configuration from our 47-config hyperparameter sweep.

#### Training

The model is trained with standard cross-entropy loss: given an abstract, the correct detailed field should have the highest logit. Training hyperparameters:

- **Learning rate:** 3e-5 (with linear warmup over 10% of training, then linear decay)
- **Epochs:** 8 passes over the training data
- **Batch size:** 16 abstracts per gradient step
- **Training data:** 23,303 abstracts (synthetic + silver-labeled real) that have a `detailed_field` annotation

After 8 epochs, the model achieves 90.3% validation accuracy on the 315-class task.

#### Inference: How a Prediction is Made

At inference time, a single forward pass through the model produces 315 logits. The prediction pipeline then works in three steps:

**Step 1 — Softmax over detailed fields:**

Apply softmax to convert the 315 raw logits into a probability distribution:

$$P(\text{detailed}_j) = \frac{e^{z_j}}{\sum_{k=1}^{315} e^{z_k}}$$

This gives us a probability for every detailed field (e.g., P("Computer Engineering, General") = 0.12, P("Electrical Engineering") = 0.08, etc.). These sum to 1.0.

**Step 2 — Marginalize to get major-field probabilities:**

The CIP taxonomy maps every detailed field to exactly one major field. To get the probability of a major field, we simply sum the probabilities of all its children:

$$P(\text{major}_m) = \sum_{j \in \text{children}(m)} P(\text{detailed}_j)$$

For example, if "Electrical and computer engineering" (a major field) has 10 detailed children, we sum those 10 probabilities to get the major-field probability. This is mathematically principled — it's exactly what probability theory says you should do to "marginalize out" the detailed variable.

This gives us 74 major-field probabilities that also sum to 1.0. No separate model needed.

**Step 3 — Strategy C scoring (combined probability):**

Finally, we score each detailed field by multiplying its own probability with its parent major field's marginalized probability:

$$\text{score}(j) = P(\text{major}_{m(j)}) \times P(\text{detailed}_j)$$

The predicted detailed field is the one with the highest score. The predicted major field is its parent in the taxonomy. The predicted broad field is its grandparent.

**Why does Strategy C help?** It acts as a soft consistency constraint. If the model assigns small probabilities to many detailed fields within the same major field (e.g., several Physics subfields each get 5%), those probabilities aggregate into a high major-field probability (e.g., Physics = 30%). Strategy C then boosts all Physics detailed fields by that 30% factor, reinforcing coherent predictions.

Conversely, if a single detailed field has high probability but belongs to an otherwise-unlikely major field, its score gets penalized by the low marginalized major probability. This reduces spurious confident predictions in isolation.

#### Concrete Example

Suppose an abstract about quantum computing produces these top detailed-field probabilities:

| Detailed Field | P(detailed) | Parent Major | P(major) | Score |
|----------------|------------|--------------|----------|-------|
| Computer Engineering, General | 0.15 | Electrical and computer eng. | 0.28 | 0.042 |
| Quantum Computing | 0.12 | Computer science | 0.35 | 0.042 |
| Computer Science, General | 0.11 | Computer science | 0.35 | 0.039 |
| Physics, General | 0.10 | Physics | 0.18 | 0.018 |

In this case, "Quantum Computing" and "Computer Engineering, General" tie on score. The model would pick whichever is marginally higher after full precision. Note how Physics—despite having a reasonably high P(detailed) of 0.10—gets suppressed because the overall P(major=Physics) is only 0.18.

#### Summary

- **One model.** SciBERT with 315-class classification head.
- **One forward pass.** Produces all 315 detailed probabilities in ~1.5s on CPU (or ~0.3s with ONNX).
- **Three prediction levels for free.** Detailed (315), major (74 via marginalization), and broad (22 via further marginalization) — all from the same 315 logits.
- **Strategy C.** Multiplies each detailed prob by its parent's marginalized major prob, boosting internally-consistent predictions.

---

## 2. Data Pipeline

### 2.1 Synthetic Training Data

**Source:** DeepSeek-R1 (671B, FP8) on Vista GH200 × 9 nodes via vLLM + Ray

Each CIP program in the taxonomy gets multiple synthetic abstracts generated with field-specific prompts. Each abstract is post-processed (strip `<think>` blocks, preamble/postamble, reject meta-language) and adversarially verified against sibling fields to ensure it's distinctive.

| Dataset | Records | Purpose |
|---------|---------|---------|
| Original synthetic (train) | 16,183 | Base training data (~10 per CIP program) |
| Synthetic test | 4,054 | Held-out evaluation set (80/20 stratified split) |

### 2.2 Real TACC Data

| Dataset | Records | Notes |
|---------|---------|-------|
| Real TACC abstracts | 16,209 | 46 of 74 major fields represented; DB labels have ~55% noise |
| UNASSIGNED records | 419 | No label — excluded from all evaluation |

### 2.3 Silver Labels (Phase C3)

Built from B4 (SetFit) + B5 (SciBERT) model consensus at confidence ≥ 0.7. Both models must agree on the major field, and at least one must have confidence ≥ 0.7.

| Dataset | Records | DB Agreement |
|---------|---------|-------------|
| Silver labels (major-level) | 5,327 | 51.2% match DB (rest are likely DB errors) |

### 2.4 Phase D2: Additional Data

| Dataset | Records | Purpose |
|---------|---------|---------|
| Targeted synthetic (14 weak fields) | 2,965 | Balanced generation: `--target-per-field 300` per major field |
| Detailed silver labels | 4,155 | Hierarchical model predictions on real TACC at high confidence |

**Combined D2 training set:** `train_d2.jsonl` — 28,630 records total (23,303 with `detailed_field`).

| Source | Count |
|--------|-------|
| Original synthetic | 16,183 |
| Major-level silver labels | 5,327 |
| Detailed silver labels | 4,155 |
| Targeted synthetic | 2,965 |

---

## 3. Evaluation Framework

### 3.1 Metrics

All approaches produce standardized `PredictionSet` objects evaluated uniformly:

- **Accuracy** at major (74), broad (22), and detailed (315) field levels
- **Macro F1** — penalizes poor performance on rare classes
- **Top-k accuracy** (k=3, 5) — correct field in top-k predictions
- **Per-field F1** — identifies systematically weak classes
- **Confusion pairs** — top misclassification patterns

### 3.2 DB Label Noise

The TACC database labels are **not reliable ground truth**. An audit of 15,790 labeled records (Phase C2) found:

- B4 + B5 models agree on major field for 7,557 records (47.9%)
- When models agree, only 44.6% match the DB label — **55.4% disagree**
- Three fields have 100% model-vs-DB disagreement when models agree: Computer Science (n=860), Mechanical Engineering (n=933), Biological Sciences General (n=566)

**Conclusion:** The classifier is likely more accurate than existing labels. DB accuracy metrics understate true model performance.

### 3.3 Three Evaluation Strategies

| Strategy | Method | Use |
|----------|--------|-----|
| **Filtered** | All records except UNASSIGNED (n=15,790) | Baseline, but unreliable |
| **Clean subset** | Records where B4, B5, and DB all agree (n=3,371) | High-confidence ground truth |
| **Consensus** | Agreement with B4/B5 consensus at conf ≥ 0.7 (n=5,327) | Model-vs-model comparison |

The C4 best model achieves **96.7% on the clean subset** and **99.6% consensus agreement**.

---

## 4. Model Development

### 4.1 Phase B: Baseline Approaches

Seven approaches evaluated end-to-end. All trained on 16,183 synthetic abstracts only.

| # | Approach | Synth Major | Synth F1 | Real TACC Major | Status |
|---|----------|-------------|----------|-----------------|--------|
| B0 | FAISS baseline (CIP definitions) | 52.2% | 0.436 | **34.3%** | Reference |
| B1 | kNN on synthetic abstracts | 85.6% | 0.850 | 25.9% | Archived |
| B2 | TF-IDF + Logistic Regression | 82.2% | 0.840 | 24.9% | Archived |
| B3 | Embedding head (frozen + MLP) | 88.6% | 0.870 | 19.6% | Archived |
| B4 | SetFit (bge-base contrastive) | 91.9% | 0.845 | 25.8% | Used for silver labels |
| B5 | SciBERT fine-tune | 90.8% | 0.843 | 32.3% | **Primary model** |
| B6 | Zero-shot LLM (32B distill) | 15.7% | 0.306 | 31.3% | Archived |

#### Key Findings

1. **Inverse correlation:** Higher synthetic accuracy → lower real accuracy. Models overfit to synthetic style.
2. **B0 FAISS beats all trained models on real data** (34.3%). CIP definitions are actually better retrieval anchors than synthetic abstracts for real research proposals.
3. **Same fields fail everywhere:** Mechanical Engineering (n=933) gets 0% F1 across all approaches — the abstracts describe physics/simulation, not engineering identity.
4. **Full fine-tuning transfers best:** B5 (SciBERT) partially escapes the inverse pattern because full encoder adaptation captures generalizable language features.

### 4.2 Phase B Analysis: Why B0 Wins on Real Data

Six hypotheses were investigated:

| Hypothesis | Explanation |
|-----------|-------------|
| **A. Synthetic artifacts** | LLM-generated abstracts use stereotypical jargon and templates that don't appear in real proposals |
| **B. Taxonomy mismatch** | CIP taxonomy organizes by degree program (what a field *teaches*), not by research topic (what researchers *do*) |
| **C. Index diversity** | 2,355 CIP definitions span each field's breadth; 16K synthetic abstracts cluster narrowly within each field |
| **D. Majority vote robustness** | Top-10 CIP definitions naturally vote across multiple subtopics within a field |
| **E. Label noise** | Real TACC labels may be self-reported, department-assigned, or stale — capping achievable accuracy |
| **F. Encoder size** | B4 used bge-base (110M) vs B0/B1 using bge-large (335M); partially explains B4's gap |

**Actionable insight:** The problem is training distribution, not model capacity. The path forward was to improve the training signal (silver labels from real data), not to build bigger models.

### 4.3 Phase C: Closing the Synthetic→Real Gap

#### C2. Label Quality Audit

Quantified DB label noise at ~55%. Established the three evaluation strategies described in §3.3. Determined that "real TACC accuracy" metrics are unreliable.

#### C3. Silver Labels + Semi-Supervised Retraining

Added 5,327 pseudo-labeled real abstracts to training. SciBERT retrained on 21,510 combined records.

| Metric | Synth Only (B5) | + Silver Labels (C3) | Delta |
|--------|----------------|---------------------|-------|
| Major Acc | 90.8% | 92.8% | +2.0% |
| Macro F1 | 0.843 | 0.905 | +6.2% |

SetFit bge-large was also tried but underperformed (90.6% major, 0.780 F1). SciBERT fine-tuning became the sole strategy going forward.

#### C4. Hyperparameter Sweep

47 configurations swept across 4 models, learning rates, epoch counts, schedulers, regularization, and layer freezing. Runtime: ~3.5 hours on a single GH200.

| Rank | Config | Major Acc | Macro F1 |
|------|--------|-----------|----------|
| 1 | **SciBERT lr=3e-5, 8ep, freeze8** | **93.96%** | **0.937** |
| 2 | SciBERT lr=3e-5, 5ep, cosine, ls=0.05 | 93.86% | 0.928 |
| 3 | DeBERTa-v3-large lr=1e-5, 8ep | 93.78% | 0.935 |

Key findings: lr=3e-5 optimal, 8 epochs optimal (10 overfits), freezing first 8 encoder layers helps, DeBERTa-v3-large competitive but 10× slower.

#### C5. Detailed-Field Classification & Hierarchical Inference

**C5a. Flat detailed model:** SciBERT on 315 classes → 87.94% detailed accuracy, 69.92% macro F1. Long tail problem: 141 of 315 classes have <10 training samples.

**C5b. Hierarchical constrained decoding:** Combined C4 major-field model + C5a detailed-field model at inference. Four strategies evaluated:

| Strategy | Description | Detailed Acc | Major Acc |
|----------|-------------|-------------|-----------|
| A: Top-1 mask | Mask to children of top-1 major | 88.58% | 93.96% |
| B: Top-k max | Max prob across top-3 major | 54.66% | 75.84% |
| **C: Combined** | **major_prob × detailed_prob** | **88.75%** | **94.06%** |
| D: Weighted | major_prob² × detailed_prob | 88.62% | 94.01% |

Strategy C wins: both levels improve simultaneously.

### 4.4 Phase D2: Targeted Generation + Single Unified Model

#### Data Improvements

**Targeted generation (D2.1):** Generated 2,965 additional abstracts for the 14 lowest-F1 fields using `--target-per-field 300` to avoid class imbalance (a flat 100/CIP would have given Engineering Technologies 6,900 samples vs 200 for smaller fields).

**Detailed silver labels (D2.2):** Ran the hierarchical model on real TACC abstracts with separate thresholds (major ≥ 0.90, detailed ≥ 0.50), producing 4,155 records with detailed-field annotations.

#### D2a. Two-Model SciBERT Retrained

Retrained major-field SciBERT on `train_d2.jsonl` (28.6K records) with the same best hyperparams (lr=3e-5, 8ep, freeze8).

| Dataset | Major Acc | Macro F1 | Top-5 |
|---------|-----------|----------|-------|
| Synthetic test | 92.92% | 0.927 | 99.90% |
| Real TACC | 32.30% | 0.184 | 55.00% |

#### D2b. Hierarchical Combined

| Dataset | Major Acc | Macro F1 |
|---------|-----------|----------|
| Synthetic test | 94.06% | 0.936 |

Still the best approach on the synthetic test metric.

#### D2c. Single Unified Model (315 classes + marginalization)

Single SciBERT trained on 23,303 records with `detailed_field`. At inference, major-field probabilities are obtained by marginalizing (summing) detailed logits within each major field. Strategy C then combines marginalized major probs with detailed probs.

| Metric | Single Model | Two-Model Hierarchical | Delta |
|--------|-------------|----------------------|-------|
| Detailed accuracy | 87.32% | 88.75% | −1.4% |
| Major accuracy | 93.00% | 94.06% | −1.1% |
| Detailed macro F1 | 0.677 | 0.713 | −3.6% |
| Major macro F1 | 0.926 | 0.936 | −1.0% |

The single model learning curve (62.8% → 90.3% val accuracy over 8 epochs) was still climbing — more epochs or additional data would likely close the gap further.

#### D2 Target Field Improvement

The targeted data generation measurably improved the 15 weakest fields:

| Field | Pre-D2 F1 | D2 F1 | Δ |
|-------|-----------|-------|---|
| Mechanical engineering | 0.50 | 0.86 | **+0.36** |
| Public health | 0.52 | 0.70 | **+0.18** |
| Clinical psychology | 0.86 | 1.00 | **+0.14** |
| Science and math education | 0.52 | 0.65 | +0.13 |
| Materials and mining eng. | 0.86 | 0.96 | +0.10 |
| Nursing and nursing science | 0.60 | 0.67 | +0.07 |
| Psychology, other | 0.89 | 0.95 | +0.06 |
| Interdisciplinary CS | 0.88 | 0.93 | +0.05 |
| Electrical and comp. eng. | 0.75 | 0.79 | +0.04 |
| Multidisciplinary/interdisciplinary | 0.86 | 0.87 | +0.01 |
| Materials sciences | 0.93 | 0.93 | = |
| Engineering technologies | 0.62 | 0.59 | −0.03 |
| Pharmacy and pharma. sci. | 0.67 | 0.63 | −0.04 |
| Science-related technologies | 0.87 | 0.80 | −0.07 |
| **Average (15 target fields)** | **0.69** | **0.76** | **+0.07** |

Overall macro F1 improved from 0.905 → 0.927 (+0.022) across all 74 fields.

#### Decision: Single Unified Model for Deployment

**Selected: D2c (single model, Strategy C marginalization)**

Rationale:
- Only 1–1.5% behind hierarchical on all metrics
- **One model** to serve, not two — simpler deployment, fewer failure modes
- **One forward pass** at inference — faster, lower latency
- Major-field predictions are derived via marginalization, not a separate model — mathematically principled
- The 315-class model inherently provides detailed-field predictions that the two-model major-only approach cannot
- Learning curve shows the model was still improving — further training or data will close the remaining gap

Model: `output/models/single_model/`

---

## 5. Summary Comparison

### All Approaches (Synthetic Test)

| # | Approach | Major Acc | Broad Acc | Macro F1 |
|---|----------|-----------|-----------|----------|
| B0 | FAISS baseline (CIP defs) | 52.2% | 55.6% | 0.436 |
| B4 | SetFit (bge-base) | 91.9% | 92.7% | 0.845 |
| B5 | SciBERT fine-tune (synth only) | 90.8% | 91.7% | 0.843 |
| C3 | SciBERT + silver labels (5ep) | 92.8% | 93.5% | 0.905 |
| **C4** | **SciBERT sweep best (8ep, freeze8)** | **94.0%** | **94.5%** | **0.937** |
| D2a | SciBERT + D2 data (two-model) | 92.9% | 93.3% | 0.927 |
| D2b | Hierarchical (D2 major × detailed) | 94.1% | 94.6% | 0.936 |
| **D2c** | **Single model (Strategy C)** | **93.0%** | **93.5%** | **0.926** |

### Progression of Improvements

```
B5 synth-only (90.8%) → C3 +silver (92.8%) → C4 +sweep (94.0%) → D2b hierarchical (94.1%)
                                                                  → D2c single model (93.0%) ← deployed
```

---

## 6. Deployment Plan

### 6.1 FastAPI Server

**Endpoint:** `POST /classify`
```json
// Request
{"abstract": "...", "top_k": 3}

// Response
{
  "predictions": [
    {
      "detailed_field": "Computer Engineering, General",
      "major_field": "Electrical and computer engineering",
      "broad_field": "Engineering",
      "confidence": 0.94
    }
  ]
}
```

Additional endpoints:
- `GET /health` — Health check
- `POST /classify/batch` — Accepts list of abstracts

### 6.2 Web UI

Single-page app served by FastAPI at `/`:
- Paste abstract → see top-3 predictions with confidence bars
- Show broad → major → detailed hierarchy in results
- Responsive design, no authentication (internal tool)

### 6.3 Batch Classification

```bash
# Classify abstracts from file
python scripts/classify_batch.py --input abstracts.csv --output predictions.jsonl

# Or via API
curl -X POST https://vm.tacc.utexas.edu/classify/batch \
  -H "Content-Type: application/json" \
  -d @abstracts.json
```

### 6.4 Infrastructure

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model format | ONNX export | 3–5× faster CPU inference (~0.3s vs ~1.5s per abstract) |
| Compute | CPU-only VM | SciBERT is small enough; avoids GPU scheduling complexity |
| Model storage | Baked into Docker image | Simpler than artifact storage for a single small model |
| Repo structure | Separate deployment repo | Keeps ML experimentation separate from production code |

### 6.5 Docker + CI/CD

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt
COPY src/ src/
COPY model/ model/
COPY app/ app/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# GitLab CI
stages: [test, build, deploy]

build:
  stage: build
  script:
    - docker build -t cip-classifier .
    - docker push $REGISTRY/cip-classifier:$CI_COMMIT_SHORT_SHA

deploy:
  stage: deploy
  script:
    - ssh $VM_HOST "docker pull $REGISTRY/cip-classifier:$CI_COMMIT_SHORT_SHA"
    - ssh $VM_HOST "docker-compose up -d"
  only: [main]
```

---

## 7. Key Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | DB labels are not ground truth | ~55% noise rate; evaluate via inter-model consensus instead |
| 2 | Silver labels via model consensus | B4+B5 agreement at high confidence, not single-model confidence |
| 3 | SciBERT over SetFit/DeBERTa | Consistently best accuracy-to-speed tradeoff; DeBERTa-large 10× slower for <0.2% gain |
| 4 | Freeze first 8 encoder layers | C4 sweep winner — regularizes training, prevents overfitting on small data |
| 5 | Single unified model for deployment | D2c: 1% behind hierarchical but simpler (one model, one forward pass, principled marginalization) |
| 6 | Targeted generation with per-field caps | `--target-per-field 300` prevents class imbalance from flat per-CIP sampling |
| 7 | Strategy C (combined probability) | major_prob × detailed_prob outperforms top-1 masking, top-k max, and weighted variants |
| 8 | ONNX export for production | 3–5× CPU speedup; SciBERT is small enough for CPU-only deployment |

---

## 8. Charts & Artifacts

### Generated Charts

| Chart | Path |
|-------|------|
| Learning curve (D2 two-model) | `output/charts/learning_curve_slurm-finetune-797974.png` |
| Learning curve (D2 single model) | `output/charts/learning_curve_slurm-single-model-797829.png` |
| Per-field F1 comparison (multi-model) | `output/charts/per_field_f1_comparison.png` |
| Per-field F1 bottom 20 | `output/charts/per_field_f1_bottom20.png` |
| Model comparison bars | `output/reports/reports/comparison_chart.png` |
| Confusion matrices | `output/reports/reports/confusion_*.png` |
| Confidence distribution | `output/charts/confidence_distribution.png` |
| Approach progression | `output/charts/approach_progression.png` |
| Data scaling curve | `output/charts/data_scaling_curve.png` |
| Sweep leaderboard | `output/charts/sweep_leaderboard.png` |
| LR × epochs heatmap | `output/charts/lr_epochs_heatmap.png` |

### Model Artifacts

| Model | Path | Use |
|-------|------|-----|
| **Single unified (deployment)** | `output/models/single_model/` | **Production model** — 315 classes + marginalization |
| Major-field sweep best (C4) | `output/sweep/models/scibert_..._freeze8/` | Reference: best major-field-only model |
| Detailed-field flat (C5a) | `output/models/detailed_finetune/` | Reference: used in hierarchical inference |
| D2 retrained finetune | `output/models/finetune/` | D2a two-model major classifier |

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_single_model.py` | Train single 315-class model + evaluate with marginalization |
| `scripts/generate_targeted.py` | Generate targeted abstracts for weak fields (`--target-per-field`) |
| `scripts/generate_detailed_silver.py` | Generate detailed-level silver labels from hierarchical model |
| `scripts/generate_charts.py` | Generate all charts (`--slurm-log`, `--per-field-compare`) |
| `scripts/sweep_finetune.py` | Hyperparameter sweep |
| `scripts/run_hierarchical_inference.py` | Two-model hierarchical inference |
| `scripts/build_silver_labels.py` | Build major-level silver labels from model consensus |
| `scripts/audit_labels.py` | Analyze DB label noise |
