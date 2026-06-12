from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from dotenv import load_dotenv

from financial_agentic_rag.config import load_yaml


DEFAULT_MODEL_CONFIG = "configs/model_config.yaml"


class LLMClientError(RuntimeError):
    """Raised when the local LLM endpoint cannot be used."""


def _llm_config(config_path: str = DEFAULT_MODEL_CONFIG) -> dict[str, Any]:
    load_dotenv()
    config = load_yaml(config_path).get("llm", {})
    return {
        "base_url": os.getenv("OPENAI_BASE_URL", config.get("base_url", "http://127.0.0.1:8000/v1")),
        "api_key": os.getenv("OPENAI_API_KEY", config.get("api_key", "EMPTY")),
        "model": os.getenv("MODEL_NAME", config.get("model", "Qwen3-14B")),
        "temperature": float(os.getenv("MODEL_TEMPERATURE", config.get("temperature", 0.1))),
        "max_tokens": int(os.getenv("MODEL_MAX_TOKENS", config.get("max_tokens", 4096))),
    }


def chat_completion(
    messages: list[dict[str, str]],
    config_path: str = DEFAULT_MODEL_CONFIG,
) -> str:
    """Call a local OpenAI-compatible vLLM endpoint."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMClientError("Missing dependency 'openai'. Run: pip install -r requirements.txt") from exc

    config = _llm_config(config_path)
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=messages,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
        )
    except Exception as exc:
        raise LLMClientError(
            f"Failed to call local vLLM endpoint {config['base_url']} with model {config['model']}."
        ) from exc

    content = response.choices[0].message.content if response.choices else ""
    return content or ""


def stream_chat_completion(
    messages: list[dict[str, str]],
    config_path: str = DEFAULT_MODEL_CONFIG,
) -> Iterator[str]:
    """Stream text deltas from a local OpenAI-compatible vLLM endpoint."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMClientError("Missing dependency 'openai'. Run: pip install -r requirements.txt") from exc

    config = _llm_config(config_path)
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    try:
        stream = client.chat.completions.create(
            model=config["model"],
            messages=messages,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            stream=True,
        )
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content
    except Exception as exc:
        raise LLMClientError(
            f"Failed to stream local vLLM endpoint {config['base_url']} with model {config['model']}."
        ) from exc
