"""Synthetic abstract generation using a large LLM (DeepSeek-R1).

This module connects to a vLLM inference server and generates
realistic research abstracts for each major field in the CIP taxonomy.

Usage:
    # Start the inference server first (see slurm/launch_server.sbatch)
    python -m cip_classifier generate -c config/default.yaml -c config/generate.yaml
    python -m cip_classifier split -c config/default.yaml -c config/generate.yaml
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any

import httpx

from .config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(
    major_field: str,
    broad_field: str,
    detailed_fields: list[str],
    focus_area: str,
    template: str,
) -> str:
    """Build a generation prompt for a specific major field."""
    detailed_str = "\n".join(f"  - {df}" for df in detailed_fields)
    return template.format(
        broad_field=broad_field,
        major_field=major_field,
        detailed_fields=detailed_str,
        focus_area=focus_area,
    )


def load_taxonomy(cfg: PipelineConfig, project_root: Path) -> dict[str, Any]:
    """Load taxonomy and build a lookup structure by major field.

    Returns:
        Dict mapping major_field -> {broad_field, detailed_fields: [...]}
    """
    taxonomy_path = cfg.resolve_path(cfg.paths.taxonomy_json, project_root)
    with open(taxonomy_path) as f:
        records = json.load(f)

    taxonomy: dict[str, dict] = {}
    for rec in records:
        major = rec.get("Major_Field_label", "").strip()
        broad = rec.get("Broad_Field_label", "").strip()
        detailed = rec.get("Detailed_Field_label", "").strip()
        if not major:
            continue
        if major not in taxonomy:
            taxonomy[major] = {"broad_field": broad, "detailed_fields": []}
        if detailed and detailed not in taxonomy[major]["detailed_fields"]:
            taxonomy[major]["detailed_fields"].append(detailed)

    logger.info("Loaded taxonomy: %d major fields", len(taxonomy))
    return taxonomy


# ---------------------------------------------------------------------------
# Server communication
# ---------------------------------------------------------------------------


async def check_server_health(base_url: str, timeout: float = 10.0) -> bool:
    """Check if the inference server is responding."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{base_url}/v1/models")
            return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


async def wait_for_server(base_url: str, max_wait: int = 600, interval: int = 15) -> None:
    """Block until the inference server is healthy or timeout."""
    logger.info("Waiting for inference server at %s ...", base_url)
    elapsed = 0
    while elapsed < max_wait:
        if await check_server_health(base_url):
            logger.info("Server is ready.")
            return
        await asyncio.sleep(interval)
        elapsed += interval
    raise RuntimeError(
        f"Inference server at {base_url} not ready after {max_wait}s"
    )


async def get_served_model_name(base_url: str) -> str:
    """Query the server to discover the actual served model name."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{base_url}/v1/models")
        resp.raise_for_status()
        data = resp.json()
        model_id = data["data"][0]["id"]
        logger.info("Served model: %s", model_id)
        return model_id


async def call_server(
    prompt: str,
    cfg: PipelineConfig,
    client: httpx.AsyncClient,
    served_model: str,
) -> str | None:
    """Send a completion request to the OpenAI-compatible API.

    Returns the generated text, or None on failure.
    """
    url = f"{cfg.generate.server_url}/v1/completions"
    payload = {
        "model": served_model,
        "prompt": prompt,
        "max_tokens": cfg.generate.max_tokens,
        "temperature": cfg.generate.temperature,
        "stop": None,
    }

    try:
        resp = await client.post(url, json=payload, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["text"]
    except (httpx.HTTPStatusError, httpx.TimeoutException, KeyError) as e:
        logger.warning("Server request failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_response(raw_text: str) -> str | None:
    """Extract the abstract from model output, stripping <think> blocks."""
    if not raw_text:
        return None

    # Remove thinking block if present
    # Pattern: <think>...</think> (possibly spanning multiple lines)
    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

    # If the model output starts mid-think (our prompt starts with <think>\n),
    # look for </think> and take everything after
    if "</think>" in raw_text:
        text = raw_text.split("</think>", 1)[1].strip()

    # Clean up common artifacts
    text = text.strip('"').strip("'").strip()

    # Remove any leading "Abstract:" or similar headers
    text = re.sub(r"^(Abstract|ABSTRACT)\s*:?\s*", "", text, flags=re.IGNORECASE)

    return text if text else None


def validate_abstract(text: str, cfg: PipelineConfig) -> bool:
    """Check if generated abstract meets quality criteria."""
    word_count = len(text.split())
    if word_count < cfg.generate.min_abstract_length:
        return False
    if word_count > cfg.generate.max_abstract_length:
        return False
    return True


# ---------------------------------------------------------------------------
# Generation orchestration
# ---------------------------------------------------------------------------


async def generate_batch(
    prompts: list[dict],
    cfg: PipelineConfig,
    semaphore: asyncio.Semaphore,
    debug_log: Path,
    served_model: str,
) -> list[dict]:
    """Generate a batch of abstracts concurrently (bounded by semaphore)."""
    results = []

    async with httpx.AsyncClient() as client:

        async def _generate_one(item: dict) -> dict | None:
            async with semaphore:
                t0 = time.time()
                raw = await call_server(item["prompt"], cfg, client, served_model)
                elapsed = time.time() - t0

                # Build debug entry
                debug_entry = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "major_field": item["major_field"],
                    "focus_area": item["focus_area"],
                    "elapsed_seconds": round(elapsed, 2),
                    "raw_response": raw[:2000] if raw else None,
                    "status": "pending",
                }

                if raw is None:
                    debug_entry["status"] = "server_error"
                    _append_debug(debug_log, debug_entry)
                    return None

                abstract = parse_response(raw)
                if abstract is None:
                    debug_entry["status"] = "parse_failed"
                    _append_debug(debug_log, debug_entry)
                    return None

                word_count = len(abstract.split())
                if not validate_abstract(abstract, cfg):
                    debug_entry["status"] = f"validation_failed (words={word_count})"
                    _append_debug(debug_log, debug_entry)
                    logger.debug(
                        "Abstract failed validation (len=%d words)",
                        word_count,
                    )
                    return None

                debug_entry["status"] = "success"
                debug_entry["word_count"] = word_count
                _append_debug(debug_log, debug_entry)

                return {
                    "abstract": abstract,
                    "major_field": item["major_field"],
                    "broad_field": item["broad_field"],
                    "focus_area": item["focus_area"],
                    "model": served_model,
                    "temperature": cfg.generate.temperature,
                }

        tasks = [_generate_one(p) for p in prompts]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result is not None:
                results.append(result)

    return results


def _append_debug(path: Path, entry: dict) -> None:
    """Append a single debug entry to the debug log."""
    with open(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def deduplicate(records: list[dict]) -> list[dict]:
    """Remove near-duplicate abstracts based on content hash."""
    seen: set[str] = set()
    unique = []
    for rec in records:
        # Normalize: lowercase, collapse whitespace
        normalized = " ".join(rec["abstract"].lower().split())
        h = hashlib.md5(normalized.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(rec)
    return unique


def save_jsonl(records: list[dict], path: Path) -> None:
    """Write records as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved %d records to %s", len(records), path)


def load_jsonl(path: Path) -> list[dict]:
    """Load records from a JSON Lines file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


async def _run_generation(cfg: PipelineConfig, project_root: Path) -> None:
    """Async generation loop."""
    # Wait for server
    await wait_for_server(cfg.generate.server_url)

    # Discover the actual model name served by the server
    served_model = await get_served_model_name(cfg.generate.server_url)

    # Load taxonomy
    taxonomy = load_taxonomy(cfg, project_root)

    # Prepare output directory
    output_dir = cfg.resolve_path(cfg.generate.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "abstracts_raw.jsonl"

    # Load existing progress (for resumption)
    existing: list[dict] = []
    if raw_path.exists():
        existing = load_jsonl(raw_path)
        logger.info("Resuming: %d abstracts already generated", len(existing))

    # Count existing per field
    existing_counts: dict[str, int] = {}
    for rec in existing:
        mf = rec["major_field"]
        existing_counts[mf] = existing_counts.get(mf, 0) + 1

    # Build prompts for fields that still need samples
    rng = random.Random(cfg.generate.seed)
    prompts: list[dict] = []

    for major_field, info in taxonomy.items():
        already_have = existing_counts.get(major_field, 0)
        needed = cfg.generate.samples_per_field - already_have
        if needed <= 0:
            continue

        detailed = info["detailed_fields"]
        for i in range(needed):
            # Rotate through detailed fields for diversity
            focus = detailed[i % len(detailed)] if detailed else major_field
            prompt_text = build_prompt(
                major_field=major_field,
                broad_field=info["broad_field"],
                detailed_fields=detailed,
                focus_area=focus,
                template=cfg.generate.prompt_template,
            )
            prompts.append({
                "prompt": prompt_text,
                "major_field": major_field,
                "broad_field": info["broad_field"],
                "focus_area": focus,
            })

    if not prompts:
        logger.info("All fields already have %d samples. Nothing to generate.",
                    cfg.generate.samples_per_field)
        return

    rng.shuffle(prompts)
    logger.info("Generating %d abstracts across %d fields ...",
                len(prompts), len(taxonomy))

    # Debug log for raw responses and diagnostics
    debug_log = output_dir / "debug_log.jsonl"
    logger.info("Debug log: %s", debug_log)

    # Generate in batches with bounded concurrency
    semaphore = asyncio.Semaphore(cfg.generate.batch_size)
    batch_size = cfg.generate.batch_size * 4  # Process in chunks for checkpointing

    all_results = list(existing)
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        results = await generate_batch(batch, cfg, semaphore, debug_log, served_model)
        all_results.extend(results)

        # Checkpoint after each batch
        save_jsonl(all_results, raw_path)
        logger.info(
            "Progress: %d / %d prompts processed, %d total abstracts",
            min(start + batch_size, len(prompts)),
            len(prompts),
            len(all_results),
        )

    # Final deduplication
    all_results = deduplicate(all_results)
    save_jsonl(all_results, raw_path)
    logger.info("Generation complete: %d unique abstracts", len(all_results))


def run(cfg: PipelineConfig, project_root: Path) -> None:
    """Generate synthetic abstracts (entry point for CLI)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(_run_generation(cfg, project_root))


def run_split(cfg: PipelineConfig, project_root: Path) -> None:
    """Split generated abstracts into train/test sets (entry point for CLI)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    output_dir = cfg.resolve_path(cfg.generate.output_dir, project_root)
    raw_path = output_dir / "abstracts_raw.jsonl"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"No generated abstracts found at {raw_path}. Run 'generate' first."
        )

    records = load_jsonl(raw_path)
    logger.info("Loaded %d records for splitting", len(records))

    # Stratified split by major field
    rng = random.Random(cfg.generate.seed)
    by_field: dict[str, list[dict]] = {}
    for rec in records:
        by_field.setdefault(rec["major_field"], []).append(rec)

    train_records: list[dict] = []
    test_records: list[dict] = []

    for field, field_recs in sorted(by_field.items()):
        rng.shuffle(field_recs)
        split_idx = max(1, int(len(field_recs) * cfg.generate.train_ratio))
        train_records.extend(field_recs[:split_idx])
        test_records.extend(field_recs[split_idx:])

    # Shuffle final sets
    rng.shuffle(train_records)
    rng.shuffle(test_records)

    save_jsonl(train_records, output_dir / "train.jsonl")
    save_jsonl(test_records, output_dir / "test.jsonl")

    logger.info(
        "Split complete: %d train, %d test across %d fields",
        len(train_records),
        len(test_records),
        len(by_field),
    )
