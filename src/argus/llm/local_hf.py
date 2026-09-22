"""Local Hugging Face backend.

Only useful if you get access to a GPU. It exists so the experiment can be repeated
without any API spend at all, and so the write-up can state that results were checked
against an open-weights model.
"""

from __future__ import annotations

import time
from typing import Any

from argus.llm.base import LLMBackend, LLMResponse, estimate_tokens


class LocalHFLLM(LLMBackend):
    """transformers-based local generation."""

    name = "local"

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        temperature: float = 0.0,
        max_tokens: int = 256,
        device: str | None = None,
        dtype: str = "auto",
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.device = device
        self.dtype = dtype
        self._pipe = None
        # Local inference has no per-token price.
        self.usage.input_price_per_m = 0.0
        self.usage.output_price_per_m = 0.0

    def _get_pipe(self):
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The local backend needs transformers and torch.\n"
                "  pip install transformers torch accelerate\n"
                "Or set ARGUS_LLM_BACKEND=mock to stay offline."
            ) from exc

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(self.model)
        mdl = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype="auto" if self.dtype == "auto" else getattr(torch, self.dtype),
            device_map=device,
        )
        self._pipe = pipeline("text-generation", model=mdl, tokenizer=tok)
        return self._pipe

    def _generate(self, prompt: str, system: str = "", **kwargs: Any) -> LLMResponse:
        pipe = self._get_pipe()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        temperature = kwargs.get("temperature", self.temperature)
        out = pipe(
            messages,
            max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            return_full_text=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        text = out[0]["generated_text"]
        if isinstance(text, list):  # chat-formatted output
            text = text[-1].get("content", "")

        return LLMResponse(
            text=str(text).strip(),
            input_tokens=estimate_tokens(system + prompt),
            output_tokens=estimate_tokens(str(text)),
            latency_ms=latency_ms,
            model=self.model,
            meta={"task": kwargs.get("task", "answer"), "backend": "local"},
        )
