"""Адаптеры LLM-провайдеров — паттерн Adapter.

Сервисы вызывают единый метод LLMClient.complete_json(system, user, schema) и
получают dict. Реализации:
  * OpenAICompatibleLLM — любой сервер с OpenAI Chat Completions API
    (Ollama, vLLM, LM Studio, OpenAI/GPT). В docker-compose — Ollama + qwen2.5:3b-instruct.
  * HFPipelineLLM — локальная модель Hugging Face transformers (text-generation).
  * NullLLM — LLM недоступна: вызывающий код использует детерминированный fallback.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


class LLMClient(ABC):
    name = "abstract"

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 800) -> str: ...

    def complete_json(self, system: str, user: str, max_tokens: int = 800) -> dict:
        raw = self.complete(system + "\nОтвечай ТОЛЬКО валидным JSON без пояснений.", user, max_tokens)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise ValueError(f"LLM returned non-JSON: {raw[:200]}")
        return json.loads(m.group(0))


class OpenAICompatibleLLM(LLMClient):
    name = "openai-compatible"

    def __init__(self, base_url: str, model: str, api_key: str = "none", timeout: float = 60, retries: int = 2) -> None:
        import httpx

        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout,
                                 headers={"Authorization": f"Bearer {api_key}"})
        self.model, self.retries = model, retries

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        body = {"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        last = None
        for attempt in range(self.retries + 1):
            try:
                r = self.http.post("/chat/completions", json=body)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(0.5 * 2 ** attempt)  # экспоненциальная пауза между повторами
        raise LLMUnavailable(str(last))


class HFPipelineLLM(LLMClient):
    name = "hf-pipeline"

    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
        from transformers import pipeline  # type: ignore

        self.pipe = pipeline("text-generation", model=model)

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        out = self.pipe([{"role": "system", "content": system}, {"role": "user", "content": user}],
                        max_new_tokens=max_tokens, do_sample=False)
        return out[0]["generated_text"][-1]["content"]


class NullLLM(LLMClient):
    name = "none"

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        raise LLMUnavailable("LLM is not configured")


def make_llm() -> LLMClient:
    if os.getenv("LLM_BASE_URL"):
        return OpenAICompatibleLLM(os.environ["LLM_BASE_URL"], os.getenv("LLM_MODEL", "qwen2.5:3b-instruct"),
                                   os.getenv("LLM_API_KEY", "none"))
    if os.getenv("LLM_HF_MODEL"):
        return HFPipelineLLM(os.environ["LLM_HF_MODEL"])
    return NullLLM()
