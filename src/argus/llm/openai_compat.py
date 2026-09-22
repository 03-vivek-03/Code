"""OpenAI-compatible backend.

Works against any endpoint that speaks the OpenAI chat completions API: OpenAI itself,
Groq, Together, OpenRouter, Cerebras, DeepInfra, or a local vLLM or Ollama server. That
matters for this project because it means the free-tier providers and the paid fallback
are the same code path with a different base URL.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from argus.llm.base import LLMBackend, LLMResponse, estimate_tokens

#: Exception type names that mean "try again, this will probably pass": a dropped
#: connection, a timeout, a 5xx from the gateway, or a rate limit. Matched by name rather
#: than imported directly so this keeps working across `openai` SDK versions that have
#: reorganised this hierarchy before.
#:
#: This distinction exists because of a real outage. A 30-hour grid died on the first
#: `APIConnectionError` ("No route to host") it hit, after a total retry budget of about
#: three seconds — 1s then 2s of backoff, nothing after the third attempt. A momentary
#: routing blip between the run host and the gateway is entirely normal over tens of
#: hours; it is not the same failure as a permanently broken request, and treating them
#: identically means the run is one network hiccup away from dying at any moment.
_TRANSIENT_ERRORS = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectError",
        "ReadTimeout",
        "ConnectTimeout",
        "RemoteProtocolError",
    }
)

#: Never worth retrying: the response will be identical on attempt two through ten, so
#: burning minutes on backoff only delays discovering the real problem.
_FATAL_ERRORS = frozenset(
    {
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "BadRequestError",
    }
)


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in _FATAL_ERRORS:
        return False
    if name in _TRANSIENT_ERRORS:
        return True
    # An HTTP status carried on the exception (the openai SDK's APIStatusError family) is
    # the fallback signal: 5xx and 429 are transient, other 4xx generally is not.
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status >= 500 or status == 429
    # Unknown exception shape: default to retrying. A request that was always going to
    # fail wastes at most one retry cycle; a connection issue misclassified as fatal would
    # go back to killing the whole grid on the first blip, which is the failure this
    # exists to prevent.
    return True

#: The model whose published rates are used to price a run served on hardware we own.
#:
#: A self-hosted 14B model on your own GPU has no per-token bill, so a dollar figure for
#: it is a fiction. It is still worth computing, because the cost comparison against
#: content-based defences — RAGuard's k+1 generator passes against this project's zero
#: extra passes — only means something if both sides are priced on the same scale. So the
#: run reports measured tokens and latency as the real cost, plus a clearly-labelled
#: notional figure at this reference model's rates.
REFERENCE_MODEL = "gpt-4o-mini"

#: Approximate US dollars per million tokens (input, output). Update before quoting any
#: figure in the write-up; providers change pricing often.
#:
#: Keys are matched as substrings of the lowercased model name, longest first, so
#: "qwen2.5:14b-instruct-q4_K_M" matches "qwen2.5" rather than falling through. It used
#: to fall through: the only Qwen key was "qwen-2.5-7b-instruct", with hyphens where the
#: Ollama-style name has a dot and a colon, so a self-hosted Qwen was silently billed at
#: gpt-4o-mini rates. With the mid-run spend guard armed at $35 that could have aborted a
#: 30-hour grid on money nobody spent.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "qwen-2.5-7b-instruct": (0.05, 0.10),
    "qwen2.5": (0.05, 0.10),
    "qwen2": (0.05, 0.10),
    "qwen3": (0.05, 0.10),
    "qwen": (0.05, 0.10),
    "mistral": (0.10, 0.30),
    "gemma": (0.05, 0.10),
    "phi": (0.05, 0.10),
    "default": (0.15, 0.60),
}


def price_for(model: str) -> tuple[float, float]:
    """Input and output price per million tokens for a model name."""
    price, _ = price_for_detailed(model)
    return price


def price_for_detailed(model: str) -> tuple[tuple[float, float], bool]:
    """Return ((input, output) price, matched_a_known_model).

    The second element is what lets the caller label a cost column notional rather than
    presenting a fallback rate as though it were the model's real price.
    """
    key = (model or "").lower()
    # Longest key first, so "qwen2.5" wins over "qwen" for a name containing both.
    for name in sorted((k for k in PRICING if k != "default"), key=len, reverse=True):
        if name in key:
            return PRICING[name], True
    return PRICING["default"], False


class OpenAICompatLLM(LLMBackend):
    """Chat-completions client with retry and backoff."""

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "",
        api_key: str = "",
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        #: Attempts allowed for an error classified as transient (see `_is_transient`),
        #: overriding `max_retries` for that case. Exponential backoff capped at
        #: `transient_backoff_cap_s` gives a worst case of a few minutes of patience for
        #: something like a dropped connection or a gateway restart, rather than the ~3
        #: seconds `max_retries=3` alone provides.
        max_retries_transient: int = 8,
        transient_backoff_cap_s: float = 60.0,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.base_url = base_url or os.environ.get("ARGUS_LLM_BASE_URL", "")
        self.api_key = api_key or os.environ.get("ARGUS_LLM_API_KEY", "")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_retries_transient = max_retries_transient
        self.transient_backoff_cap_s = transient_backoff_cap_s
        self._client = None

        (in_price, out_price), known = price_for_detailed(model)
        self.usage.input_price_per_m = in_price
        self.usage.output_price_per_m = out_price
        # Every self-hosted run is notionally priced: the tokens are real, the dollars are
        # a stand-in so the cost comparison has a common scale.
        self.usage.price_is_notional = not known or bool(self.base_url)
        self.usage.reference_model = model if known else REFERENCE_MODEL

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The openai backend needs the openai package.\n"
                '  pip install -e ".[api]"\n'
                "Or set ARGUS_LLM_BACKEND=mock to stay offline."
            ) from exc
        if not self.api_key:
            raise RuntimeError(
                "No API key. Set ARGUS_LLM_API_KEY in your .env, or use the mock backend."
            )
        kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout_s}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def _generate(self, prompt: str, system: str = "", **kwargs: Any) -> LLMResponse:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_err: Exception | None = None
        attempt = 0
        budget = self.max_retries  # fixed unless the first error turns out transient

        while attempt < budget:
            t0 = time.perf_counter()
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                choice = resp.choices[0]
                usage = getattr(resp, "usage", None)
                return LLMResponse(
                    text=(choice.message.content or "").strip(),
                    input_tokens=getattr(usage, "prompt_tokens", 0) or estimate_tokens(prompt),
                    output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    latency_ms=latency_ms,
                    model=self.model,
                    finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
                    meta={"task": kwargs.get("task", "answer"), "backend": "openai"},
                )
            except Exception as exc:  # pragma: no cover - network dependent
                last_err = exc
                transient = _is_transient(exc)
                # Extend the budget only once a transient error is actually seen, so a
                # request that fails for a real, permanent reason still fails fast.
                if transient and budget < self.max_retries_transient:
                    budget = self.max_retries_transient

                attempt += 1
                if attempt >= budget or not transient:
                    break

                backoff_cap = self.transient_backoff_cap_s if transient else 30.0
                delay = min(2.0 ** (attempt - 1), backoff_cap)
                print(
                    f"  [llm retry {attempt}/{budget}] {type(exc).__name__}: {exc} "
                    f"— waiting {delay:.0f}s before trying again",
                    file=sys.stderr,
                )
                time.sleep(delay)

        kind = "transient" if _is_transient(last_err) else "non-retryable"
        raise RuntimeError(
            f"LLM call failed after {attempt} attempt(s) ({kind} error): {last_err}"
        ) from last_err
