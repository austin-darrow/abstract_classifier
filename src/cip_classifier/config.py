"""Configuration loading and validation for CIP Classifier pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    taxonomy_excel: str = "data/raw/SEDCIP24_TACC.xlsx"
    abstracts_excel: str = "data/raw/database_abstracts.xlsx"
    taxonomy_json: str = "data/processed/SEDCIP24.json"
    broad_fields_json: str = "data/processed/broad_fields.json"
    major_fields_json: str = "data/processed/major_fields.json"
    detailed_fields_json: str = "data/processed/detailed_fields.json"
    faiss_index: str = "output/index/cip_index.faiss"
    index_metadata: str = "output/index/cip_metadata.json"
    classification_results: str = "output/results/classification_results.json"
    evaluation_report: str = "output/reports/evaluation_report.json"
    disagreements: str = "output/reports/disagreements.json"
    embeddings_npy: str = "output/index/abstract_embeddings.npy"
    visualization_plot: str = "output/reports/embedding_visualization.png"


class ModelsConfig(BaseModel):
    index_encoder: str = "shahafvl/bge-reasoner-scientific-parent-noprompt"
    query_encoder: str = "shahafvl/bge-reasoner-scientific-parent-noprompt"
    query_prefix: str = ""


class ParseConfig(BaseModel):
    sheet_name: str = "SEDCIP24"
    drop_columns: list[str] = Field(default_factory=list)


class IndexConfig(BaseModel):
    batch_size: int = 64
    text_template: str = (
        "Broad field: {Broad_Field_label} | "
        "Major field: {Major_Field_label} | "
        "Detailed field: {Detailed_Field_label} | "
        "{SED_CIPTitle}: {CIPDefinition}"
    )


class ClassifyConfig(BaseModel):
    top_k: int = 10
    batch_size: int = 32


class EvaluateConfig(BaseModel):
    confidence_thresholds: list[float] = Field(default_factory=lambda: [0.7, 0.5, 0.3])
    top_confused_pairs: int = 20


class VisualizeConfig(BaseModel):
    method: str = "umap"  # "umap" or "tsne"
    n_neighbors: int = 15
    min_dist: float = 0.1
    perplexity: float = 30.0


class GenerateConfig(BaseModel):
    model: str = "deepseek-ai/DeepSeek-R1"
    server_url: str = "http://localhost:30000"
    temperature: float = 0.6
    max_tokens: int = 4096
    samples_per_cip: int = 3
    output_dir: str = "data/generated"
    seed: int = 42
    concurrency: int = 16
    train_ratio: float = 0.8
    min_abstract_length: int = 150
    max_abstract_length: int = 600
    max_retries: int = 3
    request_timeout: float = 300.0
    sibling_fields_json: str = "data/processed/sibling_fields.json"
    use_chat_api: bool = True


class RuntimeConfig(BaseModel):
    device: str = "auto"
    batch_size: Optional[int] = None


class PipelineConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    parse: ParseConfig = Field(default_factory=ParseConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    classify: ClassifyConfig = Field(default_factory=ClassifyConfig)
    evaluate: EvaluateConfig = Field(default_factory=EvaluateConfig)
    visualize: VisualizeConfig = Field(default_factory=VisualizeConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    def resolve_path(self, path_str: str, project_root: Path) -> Path:
        """Resolve a config path relative to project root."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        return project_root / p

    def get_device(self) -> str:
        """Resolve 'auto' device to cuda/cpu."""
        if self.runtime.device != "auto":
            return self.runtime.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(*config_paths: Path) -> PipelineConfig:
    """Load and merge one or more YAML config files.

    Later files override earlier ones (deep merge at top-level keys).
    """
    merged: dict = {}
    for path in config_paths:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _deep_merge(merged, data)
    return PipelineConfig.model_validate(merged)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
