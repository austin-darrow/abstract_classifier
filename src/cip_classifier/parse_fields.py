"""Step 0: Parse CIP taxonomy from Excel into structured JSON files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .utils import save_json


def run(cfg: PipelineConfig, project_root: Path) -> None:
    """Parse taxonomy Excel and write processed JSONs."""
    input_path = cfg.resolve_path(cfg.paths.taxonomy_excel, project_root)
    print(f"Reading taxonomy from: {input_path}")

    df = pd.read_excel(input_path, sheet_name=cfg.parse.sheet_name, header=0)

    # Drop columns specified in config
    cols_to_drop = [c for c in cfg.parse.drop_columns if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    num_records = len(df)
    print(f"Read {num_records} records")

    # Write full taxonomy
    taxonomy_path = cfg.resolve_path(cfg.paths.taxonomy_json, project_root)
    taxonomy_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(taxonomy_path, orient="records", indent=2)
    print(f"Wrote taxonomy: {taxonomy_path}")

    # Write unique field-level files
    field_outputs = [
        ("Broad_Field_label", cfg.paths.broad_fields_json),
        ("Major_Field_label", cfg.paths.major_fields_json),
        ("Detailed_Field_label", cfg.paths.detailed_fields_json),
    ]
    for col, rel_path in field_outputs:
        values = sorted(df[col].dropna().unique().tolist())
        out_path = cfg.resolve_path(rel_path, project_root)
        save_json(values, out_path)
        print(f"Wrote {len(values)} unique {col} values: {out_path}")

    print("Done.")
