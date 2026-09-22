"""LLM factory."""

from __future__ import annotations

from argus.config import LLMConfig
from argus.llm.base import LLMBackend
from argus.llm.mock import MockLLM


def build_llm(config: LLMConfig | None = None, seed: int = 42) -> LLMBackend:
    """Construct a backend from config.

    The mock backend is the default deliberately: nothing in this project should require
    an API key to run.
    """
    config = config or LLMConfig()
    backend = config.backend.lower()

    if backend == "mock":
        return MockLLM(
            model=config.model or "mock-deterministic",
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            susceptibility=config.mock_susceptibility,
            seed=seed,
        )

    if backend in ("openai", "openai_compat", "api"):
        from argus.llm.openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
            max_retries=config.max_retries,
            max_retries_transient=config.max_retries_transient,
            transient_backoff_cap_s=config.transient_backoff_cap_s,
        )

    if backend in ("local", "hf", "transformers"):
        from argus.llm.local_hf import LocalHFLLM

        return LocalHFLLM(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    raise ValueError(f"unknown LLM backend '{backend}', expected mock, openai or local")
