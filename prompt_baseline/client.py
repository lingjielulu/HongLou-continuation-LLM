"""Minimal OpenAI-compatible chat completions client."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def default_base_url() -> str:
    return (
        _env_first("PROMPT_BASELINE_BASE_URL", "DEEPSEEK_BASE_URL", "OPENAI_BASE_URL")
        or "https://api.deepseek.com"
    ).rstrip("/")


def default_api_key() -> str | None:
    return _env_first("PROMPT_BASELINE_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY")


def default_model() -> str:
    return _env_first("PROMPT_BASELINE_MODEL", "DEEPSEEK_MODEL", "OPENAI_MODEL") or "deepseek-v4-pro"


def chat_completion(
    messages: list[dict[str, str]],
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.75,
    max_tokens: int = 4000,
    timeout: int = 180,
) -> str:
    key = api_key or default_api_key()
    if not key:
        raise RuntimeError(
            "未找到 API key。请设置 PROMPT_BASELINE_API_KEY、DEEPSEEK_API_KEY 或 OPENAI_API_KEY。"
        )

    endpoint = f"{(base_url or default_base_url()).rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"模型接口返回 HTTP {exc.code}: {detail}") from exc
    data = json.loads(body)
    return data["choices"][0]["message"]["content"].strip()
