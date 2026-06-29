#!/usr/bin/env python3
"""Test script to determine how vLLM handles DeepSeek-R1 chat API.

Run this against a live vLLM server serving DeepSeek-R1 to check:
1. Does /v1/chat/completions return `reasoning_content` separately from `content`?
2. Is the content clean (no <think> leakage)?
3. How does it compare to /v1/completions with manual <think> prefix?

Usage:
    python3 scripts/test_chat_api.py --server-url http://<node>:30000
"""

import argparse
import asyncio
import json
import sys

import httpx


TEST_PROMPT = (
    "Write a research abstract for a study in the following academic program.\n\n"
    "Program: Agricultural Economics\n"
    "Definition: A program that focuses on the application of economics to the "
    "analysis of resource allocation, productivity, investment, and trends in the "
    "agricultural sector, both domestically and internationally.\n"
    "Academic field: Economics (part of Social sciences)\n\n"
    "Related fields in Social sciences (for context, NOT for this abstract):\n"
    "- Political science and government\n"
    "- Sociology\n"
    "- Area, ethnic, cultural, gender, and group studies\n\n"
    "Style: empirical study\n\n"
    "Write 300-500 words in the style of a published research paper. "
    "Include background, methodology, results, and conclusions. "
    "Output ONLY the abstract text — no title, no author names, no metadata, "
    "no commentary, no markdown formatting."
)


async def test_chat_api(base_url: str, model: str) -> dict:
    """Test the chat completions endpoint."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "max_tokens": 4096,
        "temperature": 0.6,
    }

    async with httpx.AsyncClient() as client:
        print(f"\n{'='*60}")
        print("TEST: /v1/chat/completions (user message only, no system)")
        print(f"{'='*60}")
        resp = await client.post(url, json=payload, timeout=300.0)
        resp.raise_for_status()
        data = resp.json()

    choice = data["choices"][0]
    message = choice["message"]

    result = {
        "has_reasoning_content": "reasoning_content" in message,
        "reasoning_content": message.get("reasoning_content", ""),
        "content": message.get("content", ""),
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage", {}),
    }

    print(f"\nFinish reason: {result['finish_reason']}")
    print(f"Has reasoning_content field: {result['has_reasoning_content']}")
    if result["reasoning_content"]:
        rc = result["reasoning_content"]
        print(f"Reasoning content length: {len(rc)} chars")
        print(f"Reasoning preview: {rc[:200]}...")
    print(f"\nContent length: {len(result['content'])} chars")
    print(f"Content starts with '<think>': {result['content'].startswith('<think>')}")
    print(f"Content contains '</think>': {'</think>' in result['content']}")
    print(f"\n--- CONTENT ---")
    print(result["content"][:1000])
    print(f"--- END ---")

    return result


async def test_completions_api(base_url: str, model: str) -> dict:
    """Test the completions endpoint with <think> prefix."""
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model,
        "prompt": f"<think>\n{TEST_PROMPT}",
        "max_tokens": 4096,
        "temperature": 0.6,
    }

    async with httpx.AsyncClient() as client:
        print(f"\n{'='*60}")
        print("TEST: /v1/completions (with <think> prefix)")
        print(f"{'='*60}")
        resp = await client.post(url, json=payload, timeout=300.0)
        resp.raise_for_status()
        data = resp.json()

    text = data["choices"][0]["text"]
    has_think_close = "</think>" in text

    result = {
        "raw_text": text,
        "has_think_close": has_think_close,
        "finish_reason": data["choices"][0].get("finish_reason"),
        "usage": data.get("usage", {}),
    }

    print(f"\nFinish reason: {result['finish_reason']}")
    print(f"Contains </think>: {has_think_close}")
    if has_think_close:
        after_think = text.split("</think>")[-1].strip()
        print(f"Content after </think>: {len(after_think)} chars")
        print(f"\n--- CONTENT (after </think>) ---")
        print(after_think[:1000])
        print(f"--- END ---")
    else:
        print(f"Raw text length: {len(text)} chars")
        print(f"\n--- RAW (first 500) ---")
        print(text[:500])
        print(f"--- END ---")

    return result


async def get_model_name(base_url: str) -> str:
    """Get the served model name."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{base_url}/v1/models", timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["id"]


async def main(base_url: str):
    print(f"Server: {base_url}")

    # Check server health
    try:
        model = await get_model_name(base_url)
        print(f"Model: {model}")
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        sys.exit(1)

    # Run both tests
    chat_result = await test_chat_api(base_url, model)
    completions_result = await test_completions_api(base_url, model)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if chat_result["has_reasoning_content"] and chat_result["content"]:
        print("✓ Chat API separates reasoning from content — USE CHAT API")
        print("  reasoning_content is populated, content is clean")
    elif "<think>" not in chat_result["content"] and chat_result["content"]:
        print("✓ Chat API returns clean content (no <think> tags) — USE CHAT API")
        print("  (reasoning may be stripped by vLLM internally)")
    else:
        print("✗ Chat API does NOT separate thinking — USE COMPLETIONS API")
        print("  Will need manual <think> prefix + parse_response()")

    # Save full results
    output = {
        "chat_api": {
            "has_reasoning_content": chat_result["has_reasoning_content"],
            "content_starts_with_think": chat_result["content"].startswith("<think>"),
            "content_length": len(chat_result["content"]),
            "reasoning_length": len(chat_result.get("reasoning_content", "")),
        },
        "completions_api": {
            "has_think_close": completions_result["has_think_close"],
            "raw_length": len(completions_result["raw_text"]),
        },
    }
    print(f"\n{json.dumps(output, indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test vLLM DeepSeek-R1 API behavior")
    parser.add_argument("--server-url", default="http://localhost:30000",
                        help="vLLM server base URL")
    args = parser.parse_args()
    asyncio.run(main(args.server_url))
