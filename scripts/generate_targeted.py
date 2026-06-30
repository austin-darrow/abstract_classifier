"""D2.1: Generate additional synthetic abstracts for low-performing fields.

Targets the ~15 fields with F1 < 0.80 from the C4 per-field evaluation.
Uses the existing generation pipeline but filters to specific CIP programs.

Output goes to data/generated/targeted/ to avoid overwriting existing data.
After generation, run with --merge to combine with existing training data.

Usage:
    # Generate on Vista (requires vLLM server running):
    python scripts/generate_targeted.py --samples 100

    # Generate for specific fields only:
    python scripts/generate_targeted.py --samples 50 --fields "Materials sciences" "Public health"

    # Merge into combined training file (no GPU needed):
    python scripts/generate_targeted.py --merge

    # List which fields will be targeted:
    python scripts/generate_targeted.py --list-fields
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Fields with F1 < 0.80 from C4 per-field evaluation (per_field_f1.png)
DEFAULT_TARGET_FIELDS = [
    "Engineering technologies",
    "Science and mathematics education",
    "Biological, biomedical, and biosystems engineering, other",
    "Public health",
    "Mechanical engineering",
    "Materials sciences",
    "Materials and mining engineering",
    "Electrical and computer engineering",
    "Pharmacy and pharmaceutical sciences",
    "Science-related technologies",
    "Clinical psychology",
    "Nursing and nursing science",
    "Interdisciplinary computer sciences",
    "Multidisciplinary/ interdisciplinary sciences, other",
    "Psychology, other",
]


def find_project_root() -> Path:
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return cwd


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {path}")


def get_target_cips(taxonomy_path: Path, target_major_fields: list[str]) -> list[dict]:
    """Get CIP programs belonging to target major fields."""
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)

    target_set = set(target_major_fields)
    matched = [entry for entry in taxonomy if entry["Major_Field_label"] in target_set]
    matched_majors = {entry["Major_Field_label"] for entry in matched}

    # Report which fields were found / not found
    for field in target_major_fields:
        if field in matched_majors:
            n = sum(1 for e in matched if e["Major_Field_label"] == field)
            print(f"  ✓ {field}: {n} CIP programs")
        else:
            print(f"  ✗ {field}: NOT FOUND in taxonomy")

    return matched


def list_fields(project_root: Path, target_fields: list[str]):
    """Show which fields will be targeted and their current training counts."""
    taxonomy_path = project_root / "data" / "processed" / "SEDCIP24.json"
    train_path = project_root / "data" / "generated" / "train.jsonl"

    print("=== Target Fields for Data Generation ===\n")
    get_target_cips(taxonomy_path, target_fields)

    if train_path.exists():
        records = load_jsonl(train_path)
        counts = Counter(r.get("major_field", "") for r in records)
        print(f"\nCurrent training counts (from {train_path.name}):")
        for field in sorted(target_fields):
            n = counts.get(field, 0)
            print(f"  {field}: {n} samples")


def generate(project_root: Path, target_fields: list[str], samples_per_cip: int, config_paths: list[Path] | None, server_url: str | None = None, target_per_field: int | None = None):
    """Generate abstracts for target fields using existing pipeline.

    If target_per_field is set, calculates per-CIP samples dynamically for each
    field so total (existing_train + targeted) ≈ target_per_field per major field.
    Falls back to samples_per_cip if target_per_field is not set.
    """
    from cip_classifier.config import load_config
    from cip_classifier.generation.pipeline import (
        build_detailed_fields_map,
        check_server_health,
        deduplicate,
        get_served_model_name,
        load_sibling_map,
        load_jsonl as pipeline_load_jsonl,
        process_cip_program,
        save_jsonl as pipeline_save_jsonl,
        wait_for_server,
    )
    import asyncio
    import httpx
    import math

    # Load config
    if not config_paths:
        config_paths = [
            project_root / "configs" / "default.yaml",
            project_root / "configs" / "generate.yaml",
        ]
    cfg = load_config(*config_paths)
    cfg.generate.samples_per_cip = samples_per_cip
    if server_url:
        cfg.generate.server_url = server_url

    # Get target CIP programs
    taxonomy_path = project_root / "data" / "processed" / "SEDCIP24.json"
    target_cips = get_target_cips(taxonomy_path, target_fields)

    # Calculate per-field CIP counts
    cips_per_field: dict[str, int] = Counter(
        cip["Major_Field_label"] for cip in target_cips
    )

    # If target_per_field is set, calculate how many samples each field needs
    if target_per_field:
        # Load existing training data to see current counts
        train_path = project_root / "data" / "generated" / "train.jsonl"
        existing_train = load_jsonl(train_path) if train_path.exists() else []
        train_counts = Counter(r.get("major_field", "") for r in existing_train)

        samples_per_field: dict[str, int] = {}
        print(f"\nTarget per major field: {target_per_field}")
        print(f"{'Field':<55} {'Have':>5} {'Need':>5} {'CIPs':>5} {'Per CIP':>7}")
        print("-" * 82)
        for field in sorted(target_fields):
            have = train_counts.get(field, 0)
            need = max(0, target_per_field - have)
            n_cips = cips_per_field.get(field, 0)
            per_cip = math.ceil(need / n_cips) if n_cips > 0 else 0
            samples_per_field[field] = per_cip
            print(f"  {field:<53} {have:>5} {need:>5} {n_cips:>5} {per_cip:>7}")

        total_expected = sum(
            samples_per_field.get(cip["Major_Field_label"], 0) * cips_per_field.get(cip["Major_Field_label"], 0)
            for cip in {c["Major_Field_label"]: c for c in target_cips}.values()
        )
        print(f"\nExpected total new abstracts: ~{total_expected}")
    else:
        samples_per_field = None
        print(f"\nTotal CIP programs to generate for: {len(target_cips)}")
        print(f"Samples per CIP: {samples_per_cip}")
        print(f"Expected output: ~{len(target_cips) * samples_per_cip} abstracts")

    # Output to targeted/ subdirectory
    output_dir = project_root / "data" / "generated" / "targeted"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "abstracts_raw.jsonl"
    debug_path = output_dir / "debug_log.jsonl"

    # Load existing if resuming
    existing = []
    if output_path.exists():
        existing = pipeline_load_jsonl(output_path)
        print(f"Resuming: {len(existing)} existing records found")

    existing_by_cip = Counter(r.get("cip_title", "") for r in existing)

    # Filter to CIPs that still need generation
    cips_to_generate = []
    for cip in target_cips:
        field = cip["Major_Field_label"]
        per_cip = samples_per_field[field] if samples_per_field else samples_per_cip
        done = existing_by_cip.get(cip.get("SED_CIPTitle", ""), 0)
        remaining = per_cip - done
        if remaining > 0:
            cips_to_generate.extend([cip] * remaining)

    if not cips_to_generate:
        print("All target CIPs already generated. Nothing to do.")
        return

    print(f"CIP-samples remaining to generate: {len(cips_to_generate)}")

    async def _run():
        base_url = cfg.generate.server_url.rstrip("/")
        await wait_for_server(base_url)
        served_model = await get_served_model_name(base_url)
        print(f"Using model: {served_model}")

        sibling_map = load_sibling_map(cfg, project_root)
        with open(taxonomy_path) as f:
            full_taxonomy = json.load(f)
        detailed_fields_map = build_detailed_fields_map(full_taxonomy)

        semaphore = asyncio.Semaphore(cfg.generate.concurrency)
        generated = list(existing)
        completed_count = 0
        checkpoint_interval = 32  # Save every 32 completions

        async def _process_and_checkpoint(cip):
            nonlocal completed_count
            cip_title = cip.get("SED_CIPTitle", "")
            sibling_info = sibling_map.get(cip.get("Major_Field_label", ""), {})
            result = await process_cip_program(
                cip, sibling_info, cfg, semaphore, client,
                served_model, debug_path, detailed_fields_map,
            )
            if isinstance(result, dict):
                generated.append(result)
                completed_count += 1
                if completed_count % checkpoint_interval == 0:
                    pipeline_save_jsonl(generated, output_path)
                    print(f"  [Checkpoint] {completed_count} new, {len(generated)} total saved")
            return result

        async with httpx.AsyncClient(timeout=cfg.generate.request_timeout) as client:
            tasks = [_process_and_checkpoint(cip) for cip in cips_to_generate]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            errors = [r for r in results if isinstance(r, Exception)]
            if errors:
                print(f"  {len(errors)} tasks raised exceptions")
                # Show first few unique errors for debugging
                seen_types = set()
                for e in errors[:10]:
                    etype = type(e).__name__
                    if etype not in seen_types:
                        seen_types.add(etype)
                        print(f"    {etype}: {e}")

        # Final save (deduplicated)
        generated = deduplicate(generated)
        pipeline_save_jsonl(generated, output_path)

        print(f"\nGenerated {len(generated)} total targeted abstracts")
        field_counts = Counter(r.get("major_field", "") for r in generated)
        for field, count in field_counts.most_common():
            print(f"  {field}: {count}")

    asyncio.run(_run())


def merge(project_root: Path):
    """Merge all training data sources into train_d2.jsonl.

    Combines: original synthetic + major silver labels + targeted synthetic + detailed silver labels.
    Does NOT overwrite train.jsonl or train_with_silver.jsonl.
    """
    train_path = project_root / "data" / "generated" / "train.jsonl"
    silver_path = project_root / "data" / "generated" / "silver_labels.jsonl"
    targeted_path = project_root / "data" / "generated" / "targeted" / "abstracts_raw.jsonl"
    detailed_silver_path = project_root / "data" / "generated" / "detailed_silver_labels.jsonl"

    print("=== Merging Training Data (D2) ===\n")

    # Load all sources
    train_records = load_jsonl(train_path)
    for r in train_records:
        r.setdefault("source", "synthetic")
    print(f"Original synthetic train: {len(train_records)}")

    silver_records = []
    if silver_path.exists():
        silver_records = load_jsonl(silver_path)
        for r in silver_records:
            r.setdefault("source", "silver_label")
        print(f"Major-level silver labels: {len(silver_records)}")

    targeted_records = []
    if targeted_path.exists():
        targeted_records = load_jsonl(targeted_path)
        for r in targeted_records:
            r["source"] = "targeted_synthetic"
        print(f"Targeted synthetic: {len(targeted_records)}")
    else:
        print(f"No targeted data at {targeted_path} (skipping)")

    detailed_silver_records = []
    if detailed_silver_path.exists():
        detailed_silver_records = load_jsonl(detailed_silver_path)
        for r in detailed_silver_records:
            r.setdefault("source", "detailed_silver")
        print(f"Detailed-level silver labels: {len(detailed_silver_records)}")
    else:
        print(f"No detailed silver labels at {detailed_silver_path} (skipping)")

    # Combine all sources
    combined = train_records + silver_records + targeted_records + detailed_silver_records
    print(f"\nTotal combined: {len(combined)}")

    # Count records with detailed_field (usable for single-model training)
    n_with_detailed = sum(1 for r in combined if r.get("detailed_field"))
    print(f"Records with detailed_field: {n_with_detailed} ({n_with_detailed*100//len(combined)}%)")

    # Save as new file — does NOT overwrite existing
    output_path = project_root / "data" / "generated" / "train_d2.jsonl"
    save_jsonl(combined, output_path)

    # Show per-field distribution
    field_counts = Counter(r.get("major_field", "") for r in combined)
    print(f"\nField distribution ({len(field_counts)} fields):")
    for field, count in field_counts.most_common():
        print(f"  {field}: {count}")


def main():
    parser = argparse.ArgumentParser(description="D2.1: Generate targeted training data for low-F1 fields")
    parser.add_argument("--samples", type=int, default=100, help="Samples per CIP program (default: 100, ignored if --target-per-field set)")
    parser.add_argument("--target-per-field", type=int, default=None, help="Target total samples per major field (calculates per-CIP dynamically)")
    parser.add_argument("--fields", nargs="+", default=None, help="Override target fields (space-separated)")
    parser.add_argument("--config", "-c", nargs="+", type=Path, default=None, help="Config YAML file(s)")
    parser.add_argument("--server-url", type=str, default=None, help="Override vLLM server URL")
    parser.add_argument("--merge", action="store_true", help="Merge targeted data with existing training data")
    parser.add_argument("--list-fields", action="store_true", help="List target fields and current counts")
    args = parser.parse_args()

    project_root = find_project_root()
    target_fields = args.fields if args.fields else DEFAULT_TARGET_FIELDS

    if args.list_fields:
        list_fields(project_root, target_fields)
    elif args.merge:
        merge(project_root)
    else:
        generate(project_root, target_fields, args.samples, args.config, args.server_url, target_per_field=args.target_per_field)


if __name__ == "__main__":
    main()
