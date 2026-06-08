"""Synthetic abstract generation using DeepSeek-R1 via vLLM.

Pipeline v3: Per-CIP-program generation with adversarial verification.

Usage:
    # Start the inference server first (see slurm/generate_multinode.sbatch)
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
# Data loading
# ---------------------------------------------------------------------------


def load_taxonomy(cfg: PipelineConfig, project_root: Path) -> list[dict]:
    """Load the full SEDCIP24.json as a list of CIP program records."""
    taxonomy_path = cfg.resolve_path(cfg.paths.taxonomy_json, project_root)
    with open(taxonomy_path) as f:
        records = json.load(f)
    logger.info("Loaded taxonomy: %d CIP programs", len(records))
    return records


def load_sibling_map(cfg: PipelineConfig, project_root: Path) -> dict[str, Any]:
    """Load the precomputed sibling fields map.

    Returns:
        Dict mapping major_field -> {broad_field, siblings: [major_field, ...]}
    """
    path = cfg.resolve_path(cfg.generate.sibling_fields_json, project_root)
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(
    cip_title: str,
    cip_definition: str,
    major_field: str,
    broad_field: str,
    siblings: list[str],
    style: str,
    is_nec: bool = False,
) -> str:
    """Build a generation prompt for a specific CIP program.

    Follows DeepSeek-R1 guidelines: all instructions in user message, no system prompt.
    """
    if is_nec:
        return _build_nec_prompt(
            cip_title, cip_definition, broad_field, siblings, style
        )

    sibling_lines = "\n".join(f"- {s}" for s in siblings) if siblings else "- (none)"

    return (
        f"Write a research abstract for a study in the following academic program.\n\n"
        f"Program: {cip_title}\n"
        f"Definition: {cip_definition}\n"
        f"Academic field: {major_field} (part of {broad_field})\n\n"
        f"Related fields in {broad_field} (for context, NOT for this abstract):\n"
        f"{sibling_lines}\n\n"
        f"Style: {style}\n\n"
        f"Write 300-500 words in the style of a published research paper. "
        f"Include background, methodology, results, and conclusions. "
        f"Output ONLY the abstract text \xe2\x80\x94 no title, no author names, no metadata, "
        f"no commentary, no markdown formatting."
    )


def _build_nec_prompt(
    cip_title: str,
    cip_definition: str,
    broad_field: str,
    siblings: list[str],
    style: str,
) -> str:
    """Build a prompt for NEC (not elsewhere classified) fields."""
    sibling_lines = "\n".join(f"- {s}" for s in siblings)

    return (
        f"Write a research abstract for a study that falls within {broad_field} "
        f"but does NOT belong to any of the following established major fields:\n"
        f"{sibling_lines}\n\n"
        f"The research should address a topic that is clearly within {broad_field} "
        f"but occupies an interdisciplinary, emerging, or specialized niche not "
        f"covered by the fields listed above.\n\n"
        f"Program context: {cip_title}\n"
        f"Definition: {cip_definition}\n\n"
        f"Style: {style}\n\n"
        f"Write 300-500 words in the style of a published research paper. "
        f"Include background, methodology, results, and conclusions. "
        f"Output ONLY the abstract text \xe2\x80\x94 no title, no author names, no metadata, "
        f"no commentary, no markdown formatting."
    )


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

# Patterns to strip
_PREAMBLE_PATTERNS = [
    re.compile(r"^(Certainly|Sure|Of course|Here('s| is)|I('ll| will)|Let me)[^.]*[.!:]\s*", re.IGNORECASE),
    re.compile(r"^(Below is|The following is|This is)[^.]*[.!:]\s*", re.IGNORECASE),
]

_POSTAMBLE_PATTERNS = [
    re.compile(r"\n\s*(This abstract (incorporates|covers|demonstrates|highlights)[^\n]*)", re.IGNORECASE),
    re.compile(r"\n\s*(Let me know[^\n]*)", re.IGNORECASE),
    re.compile(r"\n\s*(I hope this[^\n]*)", re.IGNORECASE),
    re.compile(r"\n\s*(Note:|Keywords:|---)[^\n]*$", re.IGNORECASE | re.DOTALL),
]

_REJECT_PHRASES = [
    "the user", "this prompt", "as instructed", "as requested",
    "let me think", "first, i need to", "keywords:",
    "i'll write", "i will write", "here's a",
]

_MARKDOWN_PATTERNS = [
    re.compile(r"\*\*Abstract\*\*\s*:?\s*", re.IGNORECASE),
    re.compile(r"^#+\s*.*\n", re.MULTILINE),
    re.compile(r"\*\*(.*?)\*\*"),  # bold -> plain
    re.compile(r"---+"),
]


def postprocess(raw_text: str) -> str | None:
    """Extract and clean the abstract from model output.

    Returns cleaned abstract text, or None if the output should be rejected.
    """
    if not raw_text:
        return None

    text = raw_text

    # Handle thinking blocks: take ONLY content after </think>
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif "<think>" in text:
        # Model started thinking but never finished (hit max_tokens) -> reject
        return None

    if not text:
        return None

    # Strip markdown
    for pattern in _MARKDOWN_PATTERNS:
        text = pattern.sub(lambda m: m.group(1) if m.lastindex else "", text)

    # Strip preamble
    for pattern in _PREAMBLE_PATTERNS:
        text = pattern.sub("", text, count=1)

    # Strip postamble
    for pattern in _POSTAMBLE_PATTERNS:
        text = pattern.sub("", text)

    # Clean up quotes and whitespace
    text = text.strip().strip('"').strip("'").strip()

    # Remove leading "Abstract:" header
    text = re.sub(r"^(Abstract|ABSTRACT)\s*:?\s*", "", text, flags=re.IGNORECASE)

    # Collapse multiple newlines into paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)

    if not text:
        return None

    # Rejection checks
    text_lower = text.lower()
    for phrase in _REJECT_PHRASES:
        if phrase in text_lower:
            return None

    # Must start with capital letter
    if not text[0].isupper():
        return None

    # Must end with period (allow trailing whitespace)
    if not text.rstrip().endswith("."):
        return None

    return text.strip()


def validate_length(text: str, cfg: PipelineConfig) -> bool:
    """Check if abstract meets length criteria."""
    word_count = len(text.split())
    return cfg.generate.min_abstract_length <= word_count <= cfg.generate.max_abstract_length


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
    raise RuntimeError(f"Inference server at {base_url} not ready after {max_wait}s")


async def get_served_model_name(base_url: str) -> str:
    """Query the server to discover the actual served model name."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{base_url}/v1/models")
        resp.raise_for_status()
        data = resp.json()
        model_id = data["data"][0]["id"]
        logger.info("Served model: %s", model_id)
        return model_id


async def call_chat_api(
    prompt: str,
    cfg: PipelineConfig,
    client: httpx.AsyncClient,
    served_model: str,
) -> dict[str, str | None]:
    """Send a chat completion request. Returns {content, reasoning_content}.

    Uses only a user message (no system prompt per DeepSeek-R1 guidelines).
    """
    url = f"{cfg.generate.server_url}/v1/chat/completions"
    payload = {
        "model": served_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": cfg.generate.max_tokens,
        "temperature": cfg.generate.temperature,
    }

    try:
        resp = await client.post(url, json=payload, timeout=cfg.generate.request_timeout)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]["message"]
        return {
            "content": choice.get("content"),
            "reasoning_content": choice.get("reasoning_content"),
        }
    except (httpx.HTTPStatusError, httpx.TimeoutException, KeyError, IndexError) as e:
        logger.warning("Chat API request failed: %s", e)
        return {"content": None, "reasoning_content": None}


async def call_completions_api(
    prompt: str,
    cfg: PipelineConfig,
    client: httpx.AsyncClient,
    served_model: str,
) -> str | None:
    """Send a completions request with <think> prefix. Returns raw text."""
    url = f"{cfg.generate.server_url}/v1/completions"
    # Prepend <think>\n to enforce reasoning mode per DeepSeek-R1 guidelines
    full_prompt = f"<think>\n{prompt}"
    payload = {
        "model": served_model,
        "prompt": full_prompt,
        "max_tokens": cfg.generate.max_tokens,
        "temperature": cfg.generate.temperature,
    }

    try:
        resp = await client.post(url, json=payload, timeout=cfg.generate.request_timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["text"]
    except (httpx.HTTPStatusError, httpx.TimeoutException, KeyError, IndexError) as e:
        logger.warning("Completions API request failed: %s", e)
        return None


async def generate_one(
    prompt: str,
    cfg: PipelineConfig,
    client: httpx.AsyncClient,
    served_model: str,
) -> str | None:
    """Generate an abstract from a prompt. Returns cleaned text or None."""
    if cfg.generate.use_chat_api:
        result = await call_chat_api(prompt, cfg, client, served_model)
        # If chat API returns separated content, use it directly
        content = result.get("content")
        if content:
            return postprocess(content)
        return None
    else:
        raw = await call_completions_api(prompt, cfg, client, served_model)
        if raw is None:
            return None
        return postprocess(raw)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def build_verification_prompt(
    abstract: str,
    target_field: str,
    siblings: list[str],
    is_nec: bool = False,
) -> str:
    """Build a verification prompt to check if the abstract maps to the target field."""
    if is_nec:
        return _build_nec_verification_prompt(abstract, siblings)

    # Include target + siblings as options (shuffled)
    options = [target_field] + list(siblings)
    random.shuffle(options)
    options_str = "\n".join(f"- {o}" for o in options)

    return (
        f"Which of the following major academic fields does this abstract belong to? "
        f"Respond with ONLY the field name, exactly as written in the options.\n\n"
        f"Options:\n{options_str}\n\n"
        f"Abstract: {abstract}"
    )


def _build_nec_verification_prompt(abstract: str, siblings: list[str]) -> str:
    """Build verification prompt for NEC fields -- check it's NOT any sibling."""
    options_str = "\n".join(f"- {s}" for s in siblings)

    return (
        f"Does this abstract clearly belong to any of the following specific major fields? "
        f"If yes, state which one exactly as written. If it doesn't clearly fit any of them, "
        f'respond with exactly "none".\n\n'
        f"Options:\n{options_str}\n\n"
        f"Abstract: {abstract}"
    )


def parse_verification_response(
    response_text: str,
    target_field: str,
    is_nec: bool = False,
) -> bool:
    """Parse verification response. Returns True if verification passes."""
    if not response_text:
        return False

    cleaned = response_text.strip().strip('"').strip("'").strip()
    # Remove any thinking artifacts
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()

    if is_nec:
        # NEC passes if the verifier says "none"
        return cleaned.lower() == "none"
    else:
        # Standard field passes if verifier picks the target
        return cleaned.lower() == target_field.lower()


async def verify_abstract(
    abstract: str,
    target_field: str,
    siblings: list[str],
    is_nec: bool,
    cfg: PipelineConfig,
    client: httpx.AsyncClient,
    served_model: str,
) -> bool:
    """Run adversarial verification on a generated abstract."""
    prompt = build_verification_prompt(abstract, target_field, siblings, is_nec)

    if cfg.generate.use_chat_api:
        result = await call_chat_api(prompt, cfg, client, served_model)
        response_text = result.get("content", "")
    else:
        raw = await call_completions_api(prompt, cfg, client, served_model)
        response_text = postprocess(raw) if raw else ""

    return parse_verification_response(response_text or "", target_field, is_nec)


# ---------------------------------------------------------------------------
# Generation orchestration
# ---------------------------------------------------------------------------


async def process_cip_program(
    cip_record: dict,
    sibling_info: dict,
    style: str,
    cfg: PipelineConfig,
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    served_model: str,
    debug_log: Path,
) -> dict | None:
    """Generate and verify a single abstract for a CIP program.

    Retries up to max_retries on failure.
    """
    cip_title = cip_record["SED_CIPTitle"]
    cip_definition = cip_record["CIPDefinition"]
    major_field = cip_record["Major_Field_label"]
    broad_field = cip_record["Broad_Field_label"]
    detailed_field = cip_record["Detailed_Field_label"]
    siblings = sibling_info.get("siblings", [])
    is_nec = "nec" in major_field.lower() or "other" in major_field.lower()

    for attempt in range(1, cfg.generate.max_retries + 1):
        async with semaphore:
            t0 = time.time()

            # Build and send generation prompt
            prompt = build_prompt(
                cip_title=cip_title,
                cip_definition=cip_definition,
                major_field=major_field,
                broad_field=broad_field,
                siblings=siblings,
                style=style,
                is_nec=is_nec,
            )

            abstract = await generate_one(prompt, cfg, client, served_model)
            elapsed = time.time() - t0

            # Debug logging
            debug_entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "cip_title": cip_title,
                "major_field": major_field,
                "style": style,
                "attempt": attempt,
                "elapsed_seconds": round(elapsed, 2),
                "status": "pending",
            }

            if abstract is None:
                debug_entry["status"] = "generation_failed"
                _append_debug(debug_log, debug_entry)
                continue

            if not validate_length(abstract, cfg):
                word_count = len(abstract.split())
                debug_entry["status"] = f"length_rejected (words={word_count})"
                _append_debug(debug_log, debug_entry)
                continue

            # Verification
            verified = await verify_abstract(
                abstract, major_field, siblings, is_nec, cfg, client, served_model
            )

            debug_entry["status"] = "verified" if verified else "verification_failed"
            debug_entry["word_count"] = len(abstract.split())
            debug_entry["verified"] = verified
            _append_debug(debug_log, debug_entry)

            if not verified and attempt < cfg.generate.max_retries:
                continue

            # Return result (even if verification failed on final attempt -- record it)
            return {
                "abstract": abstract,
                "major_field": major_field,
                "broad_field": broad_field,
                "detailed_field": detailed_field,
                "cip_title": cip_title,
                "cip_definition": cip_definition,
                "style": style,
                "verified": verified,
                "attempts": attempt,
                "model": served_model,
                "temperature": cfg.generate.temperature,
            }

    return None


async def _run_generation(cfg: PipelineConfig, project_root: Path) -> None:
    """Main async generation loop."""
    # Wait for server
    await wait_for_server(cfg.generate.server_url)
    served_model = await get_served_model_name(cfg.generate.server_url)

    # Load data
    cip_programs = load_taxonomy(cfg, project_root)
    sibling_map = load_sibling_map(cfg, project_root)

    # Prepare output
    output_dir = cfg.resolve_path(cfg.generate.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "abstracts_raw.jsonl"
    debug_log = output_dir / "debug_log.jsonl"

    # Load existing progress for resumption
    existing: list[dict] = []
    if raw_path.exists():
        existing = load_jsonl(raw_path)
        logger.info("Resuming: %d abstracts already generated", len(existing))

    # Track what's already been generated: (cip_title, style) -> True
    done_keys: set[tuple[str, str]] = set()
    for rec in existing:
        done_keys.add((rec["cip_title"], rec["style"]))

    # Build work items: each CIP program x samples_per_cip styles
    rng = random.Random(cfg.generate.seed)
    styles = cfg.generate.styles

    work_items: list[tuple[dict, str]] = []
    for cip_rec in cip_programs:
        major = cip_rec["Major_Field_label"]
        if major not in sibling_map:
            logger.warning("No sibling info for major field: %s", major)
            continue

        # Assign styles (rotate through available styles)
        assigned_styles = [styles[i % len(styles)] for i in range(cfg.generate.samples_per_cip)]
        rng.shuffle(assigned_styles)

        for style in assigned_styles:
            key = (cip_rec["SED_CIPTitle"], style)
            if key not in done_keys:
                work_items.append((cip_rec, style))

    if not work_items:
        logger.info("All work complete. Nothing to generate.")
        return

    rng.shuffle(work_items)
    logger.info(
        "Generating %d abstracts (%d CIP programs x %d samples, minus %d already done)",
        len(work_items),
        len(cip_programs),
        cfg.generate.samples_per_cip,
        len(done_keys),
    )

    # Generate with bounded concurrency
    semaphore = asyncio.Semaphore(cfg.generate.concurrency)
    all_results = list(existing)
    checkpoint_interval = 64  # Save every N completions

    async with httpx.AsyncClient() as client:
        # Process in chunks for checkpointing
        for chunk_start in range(0, len(work_items), checkpoint_interval):
            chunk = work_items[chunk_start : chunk_start + checkpoint_interval]

            tasks = [
                process_cip_program(
                    cip_record=cip_rec,
                    sibling_info=sibling_map[cip_rec["Major_Field_label"]],
                    style=style,
                    cfg=cfg,
                    semaphore=semaphore,
                    client=client,
                    served_model=served_model,
                    debug_log=debug_log,
                )
                for cip_rec, style in chunk
            ]

            results = await asyncio.gather(*tasks)
            new_results = [r for r in results if r is not None]
            all_results.extend(new_results)

            # Checkpoint
            save_jsonl(all_results, raw_path)
            completed = min(chunk_start + checkpoint_interval, len(work_items))
            verified_count = sum(1 for r in all_results if r.get("verified", False))
            logger.info(
                "Progress: %d/%d items | %d total abstracts | %d verified",
                completed, len(work_items), len(all_results), verified_count,
            )

    # Final stats
    all_results = deduplicate(all_results)
    save_jsonl(all_results, raw_path)
    verified_count = sum(1 for r in all_results if r.get("verified", False))
    logger.info(
        "Generation complete: %d unique abstracts (%d verified, %d unverified)",
        len(all_results), verified_count, len(all_results) - verified_count,
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _append_debug(path: Path, entry: dict) -> None:
    """Append a single debug entry to the debug log."""
    with open(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def deduplicate(records: list[dict]) -> list[dict]:
    """Remove near-duplicate abstracts based on content hash."""
    seen: set[str] = set()
    unique = []
    for rec in records:
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
# Entry points
# ---------------------------------------------------------------------------


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
