"""Prompt construction for synthetic abstract generation."""

from __future__ import annotations


def build_prompt(
    cip_title: str,
    cip_definition: str,
    major_field: str,
    broad_field: str,
    siblings: list[str],
    is_nec: bool = False,
) -> str:
    """Build a generation prompt for a specific CIP program.

    Follows DeepSeek-R1 guidelines: all instructions in user message, no system prompt.
    """
    if is_nec:
        return _build_nec_prompt(
            cip_title, cip_definition, broad_field, siblings
        )

    sibling_lines = "\n".join(f"- {s}" for s in siblings) if siblings else "- (none)"

    return (
        f"Write a project abstract for a research computing resource allocation request "
        f"in the following academic program.\n\n"
        f"Program: {cip_title}\n"
        f"Definition: {cip_definition}\n"
        f"Academic field: {major_field} (part of {broad_field})\n\n"
        f"Other fields in {broad_field} that this abstract should NOT be about:\n"
        f"{sibling_lines}\n\n"
        f"The abstract should describe a research project requesting high-performance "
        f"computing resources. It should cover the research problem, the computational "
        f"approach or methods planned, and the expected significance or outcomes. "
        f"Write 150-400 words in a natural academic tone. The project may describe "
        f"proposed work, ongoing research, or a mix of both.\n\n"
        f"Output ONLY the abstract text \u2014 no title, no author names, no metadata, "
        f"no commentary, no markdown formatting."
    )


def _build_nec_prompt(
    cip_title: str,
    cip_definition: str,
    broad_field: str,
    siblings: list[str],
) -> str:
    """Build a prompt for NEC (not elsewhere classified) fields."""
    sibling_lines = "\n".join(f"- {s}" for s in siblings)

    return (
        f"Write a project abstract for a research computing resource allocation request "
        f"that falls within {broad_field} but does NOT belong to any of the following "
        f"established major fields:\n"
        f"{sibling_lines}\n\n"
        f"The research should address a topic that is clearly within {broad_field} "
        f"but occupies an interdisciplinary, emerging, or specialized niche not "
        f"covered by the fields listed above.\n\n"
        f"Program context: {cip_title}\n"
        f"Definition: {cip_definition}\n\n"
        f"The abstract should describe a research project requesting high-performance "
        f"computing resources. It should cover the research problem, the computational "
        f"approach or methods planned, and the expected significance or outcomes. "
        f"Write 150-400 words in a natural academic tone. The project may describe "
        f"proposed work, ongoing research, or a mix of both.\n\n"
        f"Output ONLY the abstract text \u2014 no title, no author names, no metadata, "
        f"no commentary, no markdown formatting."
    )
