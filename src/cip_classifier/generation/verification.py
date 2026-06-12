"""Adversarial verification for generated abstracts."""

from __future__ import annotations

import random

import httpx

from ..config import PipelineConfig


def build_verification_prompt(
    abstract: str,
    target_field: str,
    siblings: list[str],
    is_nec: bool = False,
    detailed_fields_map: dict[str, list[str]] | None = None,
) -> str:
    """Build a verification prompt to check if the abstract maps to the target field."""
    if is_nec:
        return _build_nec_verification_prompt(
            abstract, target_field, siblings, detailed_fields_map or {}
        )

    # Include target + siblings as options (shuffled), with detailed field context
    options = [target_field] + list(siblings)
    random.shuffle(options)

    if detailed_fields_map:
        option_lines = []
        for o in options:
            details = detailed_fields_map.get(o, [])
            if details:
                detail_str = ", ".join(details[:8])
                option_lines.append(f"- {o} (covers: {detail_str})")
            else:
                option_lines.append(f"- {o}")
        options_str = "\n".join(option_lines)
    else:
        options_str = "\n".join(f"- {o}" for o in options)

    return (
        f"Given the following academic taxonomy, which major field does this abstract "
        f"best belong to? Respond with ONLY the field name, exactly as written in the options.\n\n"
        f"Options:\n{options_str}\n\n"
        f"Abstract: {abstract}"
    )


def _build_nec_verification_prompt(
    abstract: str,
    target_field: str,
    siblings: list[str],
    detailed_fields_map: dict[str, list[str]],
) -> str:
    """Build verification prompt for NEC fields with detailed taxonomy context."""
    sibling_sections = []
    for s in siblings:
        details = detailed_fields_map.get(s, [])
        if details:
            detail_str = ", ".join(details[:8])
            sibling_sections.append(f"- {s} (covers: {detail_str})")
        else:
            sibling_sections.append(f"- {s}")
    siblings_str = "\n".join(sibling_sections)

    nec_details = detailed_fields_map.get(target_field, [])
    nec_detail_str = ", ".join(nec_details[:10]) if nec_details else target_field
    nec_section = f"- {target_field} (covers: {nec_detail_str})"

    return (
        f"Given the following academic taxonomy, which category does this abstract "
        f"best belong to?\n\n"
        f"Named major fields:\n{siblings_str}\n\n"
        f"Residual/other category:\n{nec_section}\n\n"
        f"Respond with ONLY the field name exactly as written above.\n\n"
        f"Abstract: {abstract}"
    )


def parse_verification_response(
    response_text: str,
    target_field: str,
    siblings: list[str],
    is_nec: bool = False,
) -> bool:
    """Parse verification response. Returns True if verification passes.

    Uses fuzzy matching: checks if target field appears in response,
    and that no sibling field appears instead.
    """
    if not response_text:
        return False

    cleaned = response_text.strip().strip('"').strip("'").strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()

    cleaned_lower = cleaned.lower()
    target_lower = target_field.lower()

    # Exact match (ideal case)
    if cleaned_lower == target_lower:
        return True

    # Check if target field name appears in the response
    if target_lower in cleaned_lower:
        for sib in siblings:
            if sib.lower() in cleaned_lower:
                return False
        return True

    # If the response matches a sibling exactly, it's a clear failure
    for sib in siblings:
        if cleaned_lower == sib.lower() or sib.lower() in cleaned_lower:
            return False

    # Response doesn't match target or any sibling — treat as failure
    return False


async def verify_abstract(
    abstract: str,
    target_field: str,
    siblings: list[str],
    is_nec: bool,
    cfg: PipelineConfig,
    client: httpx.AsyncClient,
    served_model: str,
    detailed_fields_map: dict[str, list[str]] | None = None,
) -> bool:
    """Run adversarial verification on a generated abstract."""
    from .pipeline import call_chat_api, call_completions_api, postprocess

    prompt = build_verification_prompt(
        abstract, target_field, siblings, is_nec, detailed_fields_map
    )

    if cfg.generate.use_chat_api:
        result = await call_chat_api(
            prompt, cfg, client, served_model, max_tokens_override=1024
        )
        response_text = result.get("content", "")
    else:
        raw = await call_completions_api(prompt, cfg, client, served_model)
        response_text = postprocess(raw) if raw else ""

    return parse_verification_response(
        response_text or "", target_field, siblings, is_nec
    )
