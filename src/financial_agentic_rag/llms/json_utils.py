from __future__ import annotations

import json
import re
from typing import Any


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def parse_json_object(text: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse an LLM JSON object, including fenced-code responses."""

    fallback = fallback or {}
    if not text:
        return fallback

    candidates = []
    block_match = JSON_BLOCK_RE.search(text)
    if block_match:
        candidates.append(block_match.group(1).strip())
    candidates.append(text.strip())

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return fallback
