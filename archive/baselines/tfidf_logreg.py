"""B2: TF-IDF + Logistic Regression baseline.

Trains in seconds, surprisingly competitive for text classification.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from ..config import PipelineConfig
from ..utils import load_json


def tfidf_classify(
    cfg: PipelineConfig,
    project_root: Path,
    train_path: Path | None = None,
    test_path: Path | None = None,
    dataset_name: str = "synthetic_test",
    max_features: int = 50000,
    ngram_range: tuple[int, int] = (1, 2),
) -> "PredictionSet":
    """Train TF-IDF + LogReg on synthetic abstracts, classify test set.

    Args:
        cfg: Pipeline config.
        project_root: Project root directory.
        train_path: Path to training JSONL.
        test_path: Path to test JSONL or Excel.
        dataset_name: Name for the prediction set.
        max_features: Max vocabulary size for TF-IDF.
        ngram_range: N-gram range for TF-IDF.

    Returns:
        PredictionSet with predictions for all test abstracts.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    from ..evaluation.predictions import Prediction, PredictionSet
    from ..baselines.faiss_retrieval import _load_jsonl, _load_test_data

    # Build major→broad mapping from taxonomy
    taxonomy_path = cfg.resolve_path(cfg.paths.taxonomy_json, project_root)
    taxonomy = load_json(taxonomy_path)
    major_to_broad = {}
    for entry in taxonomy:
        major_to_broad.setdefault(entry["Major_Field_label"], entry["Broad_Field_label"])

    # Load training data
    if train_path is None:
        train_path = cfg.resolve_path(cfg.train.train_data, project_root)
    train_records = _load_jsonl(train_path)
    print(f"Loaded {len(train_records)} training abstracts from {train_path}")

    # Load test data
    if test_path is None:
        test_path = cfg.resolve_path(cfg.train.test_data, project_root)
    test_records = _load_test_data(test_path, major_to_broad)
    print(f"Loaded {len(test_records)} test abstracts from {test_path}")

    train_texts = [r["abstract"] for r in train_records]
    train_labels = [r["major_field"] for r in train_records]
    test_texts = [r["abstract"] for r in test_records]

    # Encode labels
    le = LabelEncoder()
    le.fit(train_labels)
    y_train = le.transform(train_labels)

    # TF-IDF vectorization
    print(f"Fitting TF-IDF (max_features={max_features}, ngram_range={ngram_range})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}, X_train: {X_train.shape}")

    # Train logistic regression
    print("Training LogisticRegression (class_weight=balanced)...")
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print("Training complete.")

    # Predict with probabilities
    probs = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    pred_labels = le.inverse_transform(y_pred)

    # Build predictions
    predictions = []
    for i in range(len(test_records)):
        predicted_major = pred_labels[i]
        predicted_broad = major_to_broad.get(predicted_major, "")
        confidence = float(probs[i].max())

        # Top-k predictions
        top_k_idx = np.argsort(probs[i])[::-1][:10]
        top_k_fields = [le.classes_[idx] for idx in top_k_idx]
        top_k_scores = [float(probs[i][idx]) for idx in top_k_idx]

        predictions.append(Prediction(
            abstract=test_records[i]["abstract"],
            true_major_field=test_records[i].get("major_field", ""),
            true_broad_field=test_records[i].get("broad_field", ""),
            predicted_major_field=predicted_major,
            predicted_broad_field=predicted_broad,
            confidence=confidence,
            top_k_major_fields=top_k_fields,
            top_k_scores=top_k_scores,
        ))

    model_name = f"tfidf_logreg_{max_features}feat"
    pred_set = PredictionSet(
        model_name=model_name,
        predictions=predictions,
        dataset=dataset_name,
        metadata={
            "max_features": max_features,
            "ngram_range": list(ngram_range),
            "n_train": len(train_records),
            "n_test": len(test_records),
            "n_classes": len(le.classes_),
            "solver": "lbfgs",
            "class_weight": "balanced",
        },
    )

    print(f"TF-IDF + LogReg classification complete: {len(predictions)} predictions")
    return pred_set
