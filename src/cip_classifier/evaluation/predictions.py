"""Standardized prediction format for all classifier approaches.

Every approach (FAISS baseline, TF-IDF, embedding head, SetFit, fine-tune, LLM)
must produce predictions in this format so they can be evaluated uniformly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Prediction:
    """A single classification prediction."""

    abstract: str
    true_major_field: str
    true_broad_field: str
    predicted_major_field: str
    predicted_broad_field: str
    confidence: Optional[float] = None
    top_k_major_fields: list[str] = field(default_factory=list)
    top_k_scores: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class PredictionSet:
    """A collection of predictions from a single approach/model run."""

    model_name: str
    predictions: list[Prediction]
    dataset: str = ""  # e.g. "synthetic_test" or "real_tacc"
    metadata: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.predictions)

    def save(self, path: Path) -> None:
        """Save predictions to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_name": self.model_name,
            "dataset": self.dataset,
            "metadata": self.metadata,
            "predictions": [asdict(p) for p in self.predictions],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PredictionSet":
        """Load predictions from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        valid_keys = {f.name for f in Prediction.__dataclass_fields__.values()}
        predictions = [
            Prediction(**{k: v for k, v in p.items() if k in valid_keys})
            for p in data["predictions"]
        ]
        return cls(
            model_name=data["model_name"],
            predictions=predictions,
            dataset=data.get("dataset", ""),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_jsonl(cls, path: Path, model_name: str, dataset: str = "") -> "PredictionSet":
        """Load predictions from a JSONL file (one prediction per line)."""
        predictions = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    predictions.append(Prediction(**json.loads(line)))
        return cls(model_name=model_name, predictions=predictions, dataset=dataset)
