"""B6: Zero-shot LLM classification (accuracy ceiling).

Uses an LLM (via OpenAI-compatible API) to classify abstracts zero-shot.
Expensive but establishes the upper bound for this task.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import PipelineConfig
from ..utils import load_json


SYSTEM_PROMPT = """You are a research classifier. Given a research abstract, classify it into the most appropriate Major Field from the taxonomy below.

TAXONOMY (Major Fields grouped by Broad Field):
{taxonomy_text}

Respond with ONLY a JSON object: {{"major_field": "<exact field name>"}}
Do not include any other text."""


def _build_taxonomy_text(taxonomy: list[dict]) -> str:
    """Build a compact taxonomy reference string."""
    from collections import defaultdict
    broad_to_major = defaultdict(set)
    for entry in taxonomy:
        broad_to_major[entry["Broad_Field_label"]].add(entry["Major_Field_label"])

    lines = []
    for broad, majors in sorted(broad_to_major.items()):
        lines.append(f"\n{broad}:")
        for major in sorted(majors):
            lines.append(f"  - {major}")
    return "\n".join(lines)


def zeroshot_llm_classify(
    cfg: PipelineConfig,
    project_root: Path,
    test_path: Path | None = None,
    dataset_name: str = "synthetic_test",
    model: str | None = None,
    server_url: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    concurrency: int = 8,
    max_samples: int | None = None,
) -> "PredictionSet":
    """Classify abstracts via zero-shot LLM prompting.

    Args:
        cfg: Pipeline config.
        project_root: Project root directory.
        test_path: Path to test JSONL or Excel.
        dataset_name: Name for the prediction set.
        model: LLM model name.
        server_url: OpenAI-compatible API URL.
        temperature: Sampling temperature (0 for deterministic).
        max_tokens: Max output tokens.
        concurrency: Number of concurrent requests.
        max_samples: Cap number of samples (for cost control).

    Returns:
        PredictionSet with predictions for all test abstracts.
    """
    import asyncio
    import aiohttp

    from ..evaluation.predictions import Prediction, PredictionSet
    from .faiss_retrieval import _load_jsonl, _load_test_data

    if model is None:
        model = cfg.generate.model
    if server_url is None:
        server_url = cfg.generate.server_url

    # Build major→broad mapping
    taxonomy_path = cfg.resolve_path(cfg.paths.taxonomy_json, project_root)
    taxonomy = load_json(taxonomy_path)
    major_to_broad = {}
    valid_majors = set()
    for entry in taxonomy:
        major_to_broad.setdefault(entry["Major_Field_label"], entry["Broad_Field_label"])
        valid_majors.add(entry["Major_Field_label"])

    taxonomy_text = _build_taxonomy_text(taxonomy)
    system_prompt = SYSTEM_PROMPT.format(taxonomy_text=taxonomy_text)

    # Load test data
    if test_path is None:
        test_path = cfg.resolve_path(cfg.train.test_data, project_root)
    test_records = _load_test_data(test_path, major_to_broad)
    print(f"Loaded {len(test_records)} test abstracts")

    if max_samples is not None and max_samples < len(test_records):
        import random
        random.seed(42)
        test_records = random.sample(test_records, max_samples)
        print(f"Subsampled to {max_samples} abstracts for cost control")

    # API endpoint
    api_url = f"{server_url.rstrip('/')}/v1/chat/completions"

    async def classify_one(session: aiohttp.ClientSession, record: dict, sem: asyncio.Semaphore) -> dict:
        """Classify a single abstract via API."""
        async with sem:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Classify this abstract:\n\n{record['abstract']}"},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            for attempt in range(3):
                try:
                    async with session.post(api_url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                        if resp.status == 429:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        resp.raise_for_status()
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"].strip()
                        return {"record": record, "response": content}
                except Exception as e:
                    if attempt == 2:
                        return {"record": record, "response": "", "error": str(e)}
                    await asyncio.sleep(2 ** attempt)
        return {"record": record, "response": "", "error": "max_retries"}

    async def run_all():
        sem = asyncio.Semaphore(concurrency)
        async with aiohttp.ClientSession() as session:
            tasks = [classify_one(session, rec, sem) for rec in test_records]
            results = []
            for i in range(0, len(tasks), 100):
                batch = await asyncio.gather(*tasks[i:i+100])
                results.extend(batch)
                if i + 100 < len(tasks):
                    print(f"  Processed {i+100}/{len(tasks)}...")
            return results

    print(f"Classifying {len(test_records)} abstracts via {model} (concurrency={concurrency})...")
    results = asyncio.run(run_all())
    print(f"API calls complete. Parsing responses...")

    # Parse responses
    predictions = []
    parse_errors = 0
    for res in results:
        record = res["record"]
        response = res.get("response", "")

        # Try to parse JSON from response
        predicted_major = ""
        try:
            clean = response.strip()
            # Strip <think>...</think> reasoning blocks (DeepSeek-R1 style)
            if "<think>" in clean:
                # Take everything after the last </think>
                parts = clean.split("</think>")
                clean = parts[-1].strip() if len(parts) > 1 else clean
            # Handle markdown code block wrapping
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            # Try to find JSON object in the remaining text
            json_start = clean.find("{")
            json_end = clean.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                clean = clean[json_start:json_end]
            parsed = json.loads(clean)
            predicted_major = parsed.get("major_field", "")
        except (json.JSONDecodeError, KeyError, IndexError):
            # Try to find a field name in the response
            for mf in sorted(valid_majors, key=len, reverse=True):
                if mf.lower() in response.lower():
                    predicted_major = mf
                    break
            if not predicted_major:
                parse_errors += 1

        predicted_broad = major_to_broad.get(predicted_major, "")
        confidence = 1.0 if predicted_major in valid_majors else 0.0

        predictions.append(Prediction(
            abstract=record["abstract"],
            true_major_field=record.get("major_field", ""),
            true_broad_field=record.get("broad_field", ""),
            predicted_major_field=predicted_major,
            predicted_broad_field=predicted_broad,
            confidence=confidence,
            top_k_major_fields=[predicted_major] if predicted_major else [],
            top_k_scores=[confidence] if predicted_major else [],
        ))

    if parse_errors:
        print(f"WARNING: {parse_errors}/{len(results)} responses failed to parse")

    model_label = model.split("/")[-1] if "/" in model else model
    pred_set = PredictionSet(
        model_name=f"zeroshot_{model_label}",
        predictions=predictions,
        dataset=dataset_name,
        metadata={
            "model": model,
            "server_url": server_url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "concurrency": concurrency,
            "n_test": len(test_records),
            "parse_errors": parse_errors,
        },
    )

    print(f"Zero-shot LLM classification complete: {len(predictions)} predictions ({parse_errors} parse errors)")
    return pred_set
