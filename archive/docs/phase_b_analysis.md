Comprehensive Phase B Analysis
1. Potential Bugs and Measurement Issues
UNASSIGNED — You caught a real problem.

Looking at the code path for all approaches, the filter if p.true_major_field is applied before metrics computation for real TACC. This correctly filters out records where major_field is an empty string (NaN in Excel → empty string in _load_test_data). But "UNASSIGNED" is a non-empty string — it passes the filter and gets included as a real label.

Here's the critical asymmetry:

B0 (baseline_classify): The CLI does labeled = [p for p in pred_set_real.predictions if p.true_major_field] — same filter as everything else. So UNASSIGNED records ARE included, and B0 cannot predict "UNASSIGNED" (it's not in the taxonomy), meaning those 419 records are guaranteed wrong for every approach.
B1-B5: Same filter, same problem. But these are also trained on synthetic data that never contains "UNASSIGNED", so they also can't predict it.
B6: LLM could theoretically output anything, but we constrain it to taxonomy labels.
So UNASSIGNED is penalizing all approaches equally. It's not the source of B0's advantage. BUT it's dragging everyone's numbers down by ~2.6% (419/16209). The real accuracy should be computed on ~15,790 records.

However, there's a subtler issue: B0 reports 16,209 total predictions, meaning it evaluates on all records including those with empty labels. Wait — let me re-read the output. B0 says "Total predictions: 16209" which is the labeled set after filtering. This means there are no empty-label records — all 16,209 have a major_field value in the Excel. Some of those values are "UNASSIGNED."

Verdict: UNASSIGNED penalizes all approaches equally (~2.6% floor). Not a bug that explains B0's advantage, but we should exclude it for cleaner numbers.

Another issue: B0 uses 2,355 CIP definition vectors; B1-B4 use 16,183 synthetic abstract vectors. The FAISS indexes are completely different:

B0 index = CIP program definitions (terse, formal, definitional)
B1 index = synthetic abstracts (verbose, LLM-generated, mimicking research proposals)
This means B0's "vocabulary" (what it retrieves from) is fundamentally different. CIP definitions describe what a field is, while synthetic abstracts describe what research in that field looks like. Real TACC abstracts are research proposals — they should theoretically match synthetic abstracts better. But they don't. Why?

2. Overall Trends Across Phase B
Trend 1: Inverse correlation between synthetic accuracy and real accuracy.

Approach	Synthetic Major ↑	Real Major ↓
B0 FAISS defs	0.5224	0.3431
B2 TF-IDF	0.8219	0.2485
B1 kNN	0.8564	0.2592
B3 MLP	0.8858	0.1962
B5 SciBERT	0.9077	0.3228
B4 SetFit	0.9188	0.2579
The more a model masters synthetic data, the worse it does on real data (with B5 being a partial exception). This is textbook distribution shift overfitting.

Trend 2: B5 (full fine-tune) partially escapes the pattern. SciBERT achieves the second-best real TACC score (32.3%) despite being trained on synthetic data. This is because full fine-tuning adapts low-level language representations that partially transfer, whereas kNN/MLP/SetFit only learn decision boundaries in a fixed (or contrastively-shifted) embedding space.

Trend 3: Broad field accuracy is always higher than major field accuracy by ~15-20 percentage points. This is expected (fewer classes, coarser granularity), but it means a hierarchical approach could be very effective.

Trend 4: The same fields fail across ALL approaches. Mechanical Engineering (n=933) gets 0% F1 everywhere. "Biological and biomedical sciences, general" (n=566) is always 0%. These are structural problems:

ME abstracts describe physics/materials/simulation work — the field label is about the department, not the research topic
"Bio sciences, general" is a catch-all that gets eaten by more specific bio fields
Trend 5: Top-5 accuracy is consistently 15-20% above Top-1 for B0-B5, meaning the correct field is often in the neighborhood. A re-ranking approach could convert top-5 hits into top-1.

3. Potential Explanations for B0's Superiority
Explanation A: Synthetic abstracts have systematic linguistic artifacts.

LLM-generated abstracts follow predictable templates. DeepSeek-R1 likely:

Uses field-specific jargon more stereotypically than real researchers
Follows a "This research investigates..." structure
Over-represents certain subtopics within each field
Includes explicit field-name mentions (e.g., "this mechanical engineering study...")
Real TACC abstracts are proposals for computing allocations — they describe computational methods, not field identity. A materials scientist and a mechanical engineer might write nearly identical proposals about molecular dynamics simulation.

Explanation B: The CIP taxonomy is organized by what a degree program teaches, not by what research looks like.

This is the deepest issue. "Mechanical Engineering" as a CIP program covers turbines, robotics, HVAC, biomechanics, etc. But a real ME research abstract might describe "finite element analysis of composite materials under thermal loading" — which is indistinguishable from Materials Science or Physics.

B0 works better because CIP definitions describe the program's scope ("A program that focuses on..."), which is actually closer to how a human would categorize. The definitions contain broad scope language, not specific research jargon. Synthetic abstracts, by contrast, try too hard to sound like research in that field, which paradoxically makes them less useful as retrieval anchors.

Explanation C: Index diversity matters — 2,355 diverse definitions vs. 16,183 homogeneous synthetics.

B0's index has 2,355 CIP entries spanning every corner of each field. B1's index has ~220 abstracts per field (16,183 / 74), all generated by the same LLM. The synthetic abstracts for a field may cluster tightly in embedding space, creating "dead zones" where real abstracts fall between fields. The CIP definitions, being more diverse and spanning the full breadth of each field, may provide better coverage.

Explanation D: Majority vote over CIP definitions is more robust.

When you query the CIP index, the top-10 hits span multiple detailed fields within a major field. This gives a natural "weighted vote" where the correct major field accumulates votes from many detailed-field definitions. With synthetic abstracts, the top-10 hits might all be from the same narrow subtopic, making the vote less informative about the broader field.

Explanation E: Real TACC data may have labeling issues.

If the major_field labels in the Excel were assigned by automated tools, self-reported by PIs, or mapped from department codes, they may be:

Inconsistent (same research labeled differently depending on PI's department)
Stale (labels assigned at allocation request time, not updated)
Wrong (PI picks closest label, not necessarily correct)
This would cap accuracy for ANY approach at well below 100%.

Explanation F: Encoder choice (B4 uses bge-base, not bge-large).

B4 SetFit uses bge-base-en-v1.5 (from train.yaml) while B0/B1/B3 use bge-large-en-v1.5 (from vista.yaml). This is a smaller model (110M vs 335M, 768-dim vs 1024-dim) which may partially explain B4's weaker real-world performance. B5 uses SciBERT (110M) — also smaller but domain-specialized.

4. Additional Observations
The "label noise" problem may dominate. 419 UNASSIGNED records + potentially hundreds of mislabeled records could mean 5-10% of labels are wrong. This caps the achievable accuracy.

The Mechanical Engineering problem is diagnostic. 933 abstracts get 0% F1 across every approach. These are ~5.7% of the data. If ME is systematically mislabeled or genuinely ambiguous, it alone accounts for a ~6% hit to accuracy.

B6's performance is uninformative. The 32B distill model is not a true ceiling — it has a CS bias and taxonomy name mismatch issues. The real zero-shot ceiling would require GPT-4/Claude with careful prompt engineering and should be measured, possibly on a small hand-labeled subset.

The most promising Phase C direction is not "improve the classifier" — it's "improve the training signal":

Use B0's confident predictions as pseudo-labels to train B5 on real abstracts
Augment the FAISS index with real TACC abstracts that B0 classifies with high confidence (>0.7 agreement ratio)
Hierarchical: broad first, then major — B0 gets 60% broad accuracy, which is a much better starting signal
Manual labeling of 200-500 abstracts to establish a true ceiling and measure label noise