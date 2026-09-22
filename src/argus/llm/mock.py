"""Deterministic offline LLM backend.

Why this exists
---------------
The entire platform must be buildable, testable and demonstrable with no API key, no
GPU and no network. This backend makes that possible. It is not a stub that returns
fixed strings: it reads the retrieved context, extracts the claims that context makes,
and answers according to the balance of evidence, so the pipeline that runs here is the
same pipeline that later runs against a real model.

Behaviourally it simulates:

* susceptibility to poisoned context, scaled by how much of the context is poisoned
* reflection reducing susceptibility, because a reflecting agent gets a second look
* document inspection helping slightly, because inspection surfaces full text
* query rewriting producing genuinely different search strings
* deterministic output, seeded from the prompt, so runs are exactly reproducible

Important
---------
Numbers produced with this backend are NOT research results. They exercise the
pipeline. Every reported experiment must use a real backend. The runner records the
backend name in every trace and the analysis layer refuses to publish summaries built
on mock traces unless explicitly asked.
"""

from __future__ import annotations

import hashlib
import random
import re
import time
from typing import Any

from argus.llm.base import LLMBackend, LLMResponse, estimate_tokens

_ANSWER_PAT = re.compile(
    r"(?:the answer is|answer:|is|was|in)\s+([A-Z][\w' .-]{1,40}?)(?:\.|,|$)",
    re.IGNORECASE,
)


class MockLLM(LLMBackend):
    """A deterministic, context-sensitive simulated model."""

    name = "mock"

    def __init__(
        self,
        model: str = "mock-deterministic",
        temperature: float = 0.0,
        max_tokens: int = 256,
        susceptibility: float = 0.85,
        seed: int = 42,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.susceptibility = susceptibility
        self.seed = seed
        # Mock calls are free. Zero prices keep cost reporting honest.
        self.usage.input_price_per_m = 0.0
        self.usage.output_price_per_m = 0.0

    # ------------------------------------------------------------- determinism
    def _rng(self, prompt: str) -> random.Random:
        h = hashlib.sha256(f"{self.seed}|{prompt}".encode()).hexdigest()
        return random.Random(int(h[:16], 16))

    # ------------------------------------------------------------------ router
    def _generate(self, prompt: str, system: str = "", **kwargs: Any) -> LLMResponse:
        t0 = time.perf_counter()
        task = kwargs.get("task", self._infer_task(prompt, system))

        handlers = {
            "rewrite": self._do_rewrite,
            "reflect": self._do_reflect,
            "inspect": self._do_inspect,
            "judge": self._do_judge,
            "answer": self._do_answer,
        }
        text = handlers.get(task, self._do_answer)(prompt, kwargs)

        # Simulated latency, deterministic per prompt, so latency features are stable.
        latency = 40.0 + self._rng(prompt).random() * 60.0
        _ = time.perf_counter() - t0

        return LLMResponse(
            text=text,
            input_tokens=estimate_tokens(system + prompt),
            output_tokens=estimate_tokens(text),
            latency_ms=latency,
            model=self.model,
            meta={"task": task, "backend": "mock"},
        )

    @staticmethod
    def _infer_task(prompt: str, system: str) -> str:
        blob = f"{system} {prompt}".lower()
        if "rewrite" in blob or "reformulate" in blob or "search quer" in blob:
            return "rewrite"
        if "reflect" in blob or "is the evidence sufficient" in blob:
            return "reflect"
        if "which document" in blob or "inspect" in blob:
            return "inspect"
        if "judge" in blob or "is this document suspicious" in blob:
            return "judge"
        return "answer"

    # ------------------------------------------------------------------- tasks
    def _do_rewrite(self, prompt: str, kw: dict[str, Any]) -> str:
        """Produce a reformulated search query.

        Rewrites are lexically different but semantically anchored on the original,
        which is what makes query-drift a usable detection feature: benign drift is
        bounded, attacker-induced drift is not.
        """
        question = kw.get("question", self._extract_question(prompt))
        rng = self._rng(prompt)
        words = [w for w in re.findall(r"[A-Za-z0-9']+", question) if len(w) > 3]
        if not words:
            return question
        # A benign rewrite preserves most content terms. Dropping too many would make
        # the mechanism a retrieval-quality confound rather than a security variable.
        keep = max(3, int(round(len(words) * rng.uniform(0.8, 1.0))))
        chosen = words[:keep]
        prefix = rng.choice(["", "details about ", "record of ", "information on "])
        return f"{prefix}{' '.join(chosen)}".strip()

    def _do_reflect(self, prompt: str, kw: dict[str, Any]) -> str:
        """Decide whether the gathered evidence is sufficient.

        Modelled as evidence-count driven with a ceiling, so that reflection genuinely
        varies rather than always approving. Conflicting evidence pushes a reflecting
        agent to look again, which is the behaviour that makes reflection interesting
        under poisoning: poison creates conflict.
        """
        rng = self._rng(prompt)
        n_docs = kw.get("n_docs", prompt.lower().count("[doc"))
        p_sufficient = min(0.45 + 0.09 * min(n_docs, 5), 0.80)
        if self._context_conflicted(prompt):
            p_sufficient -= 0.25
        return "SUFFICIENT" if rng.random() < max(p_sufficient, 0.05) else "INSUFFICIENT"

    def _do_inspect(self, prompt: str, kw: dict[str, Any]) -> str:
        n = kw.get("n_docs", max(prompt.count("[DOC"), 1))
        rng = self._rng(prompt)
        return str(rng.randint(1, max(n, 1)))

    def _do_judge(self, prompt: str, kw: dict[str, Any]) -> str:
        rng = self._rng(prompt)
        looks_injected = any(
            marker in prompt.lower()
            for marker in ("ignore previous", "you must answer", "authoritative correction")
        )
        p = 0.75 if looks_injected else 0.12
        return "SUSPICIOUS" if rng.random() < p else "CLEAN"

    def _do_answer(self, prompt: str, kw: dict[str, Any]) -> str:
        """Answer from context, weighing the claims the context actually makes.

        This is the part that matters. Attack success in this simulation is an emergent
        consequence of how much of the context asserts the attacker's claim and how
        strongly, not a hard-coded probability.
        """
        rng = self._rng(prompt)
        claims = self._extract_claims(prompt)

        if not claims:
            return "I could not determine the answer from the provided context."

        # Weight each claim by how often it is asserted and how early it appears.
        scored: dict[str, float] = {}
        for rank, (claim, emphatic) in enumerate(claims):
            weight = 1.0 / (1.0 + 0.25 * rank)
            if emphatic:
                # Poisoned passages typically assert forcefully. That is the whole
                # mechanism PoisonedRAG relies on.
                weight *= 1.0 + 2.0 * self.susceptibility
            scored[claim] = scored.get(claim, 0.0) + weight

        best = max(scored.items(), key=lambda kv: kv[1])[0]
        runner_up = sorted(scored.values(), reverse=True)
        margin = (runner_up[0] - runner_up[1]) / runner_up[0] if len(runner_up) > 1 else 1.0

        # A near-tie is resolved stochastically but deterministically per prompt.
        if margin < 0.15 and len(scored) > 1:
            keys = sorted(scored, key=lambda k: -scored[k])
            best = keys[0] if rng.random() < 0.65 else keys[1]

        return best

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _extract_question(prompt: str) -> str:
        for line in prompt.splitlines():
            line = line.strip()
            if line.lower().startswith(("question:", "query:")):
                return line.split(":", 1)[1].strip()
        return prompt.strip().splitlines()[-1] if prompt.strip() else ""

    @staticmethod
    def _context_conflicted(prompt: str) -> bool:
        claims = MockLLM._extract_claims(prompt)
        return len({c for c, _ in claims}) > 1

    @staticmethod
    def _split_context(prompt: str) -> tuple[str, str]:
        """Return (context_body, question)."""
        low = prompt.lower()
        body = prompt
        if "context:" in low:
            body = prompt[low.index("context:") + len("context:") :]
        question = MockLLM._extract_question(prompt)
        if "question:" in body.lower():
            body = body[: body.lower().rindex("question:")]
        return body, question

    #: Phrasing published poisoning attacks use to override competing evidence. Kept in
    #: sync with the attack templates in `argus.attacks`; when the templates were made
    #: grammatical this list had to move with them, and a test now asserts the mock is
    #: still misled so the two cannot drift apart unnoticed.
    EMPHATIC_MARKERS = (
        "you must", "ignore", "authoritative", "definitively", "the only correct",
        "regardless of", "important:", "verified record", "corrected record",
        "supersedes", "it is now established", "standing correction",
        "independent verification", "should not be relied upon",
    )

    @staticmethod
    def _extract_claims(prompt: str) -> list[tuple[str, bool]]:
        """Simulate extractive reading comprehension over the context.

        Relevance is judged per *document*, not per sentence, and the answer span is then
        taken from whichever sentence inside a relevant document carries one.

        That distinction matters. A real passage introduces its subject in one sentence
        and states the fact in another, and so does a realistic poisoned passage: the
        attack's retrieval condition puts the query terms at the top and its generation
        condition asserts the answer further down. Scoring sentence by sentence, the
        assertion sentence shares almost no vocabulary with the question and was
        discarded, so the mock could not be poisoned by any passage whose claim was not
        in the same sentence as the query terms. That made offline poisoning tests pass
        only for one specific template shape.

        Returns (claim, emphatic) pairs ordered by relevance.
        """
        body, question = MockLLM._split_context(prompt)
        if not body.strip():
            return []

        q_terms = MockLLM._content_terms(question)
        relation = MockLLM._question_relation(question)

        # Split into document blocks, matching how the context is rendered.
        blocks = re.split(r"\[DOC \d+\]", body)
        scored: list[tuple[float, str, bool]] = []

        for block in blocks:
            block = block.strip()
            if len(block) < 8:
                continue

            b_terms = MockLLM._content_terms(block)
            overlap = len(q_terms & b_terms) / max(len(q_terms), 1)
            if relation and relation in block.lower():
                overlap += 0.5
            if overlap < 0.2:
                continue

            emphatic = any(m in block.lower() for m in MockLLM.EMPHATIC_MARKERS)

            # Within a relevant document, take the first sentence that yields a span.
            for raw in re.split(r"(?<=[.!?])\s+|\n", block):
                seg = raw.strip()
                if len(seg) < 8:
                    continue
                span = MockLLM._answer_span(seg, relation)
                if span:
                    scored.append((overlap, span, emphatic))
                    break

        scored.sort(key=lambda t: -t[0])
        return [(span, emph) for _, span, emph in scored]

    @staticmethod
    def _content_terms(text: str) -> set[str]:
        stop = {
            "the", "a", "an", "was", "were", "is", "are", "in", "of", "to", "and", "that",
            "it", "its", "what", "which", "who", "for", "by", "on", "at", "from", "with",
            "according", "official", "record", "documented", "sources", "agree", "this",
        }
        toks = re.findall(r"[a-z0-9']+", text.lower())
        return {t for t in toks if t not in stop and len(t) > 2}

    @staticmethod
    def _question_relation(question: str) -> str:
        """Pull the relation phrase out of the question, e.g. 'was founded in'.

        The synthetic and real question formats both put the relation immediately before
        the interrogative, so this is a reliable anchor for the answer span.
        """
        m = re.search(
            r"\b((?:was|is|were|are|has|have)?\s*[a-z]+(?:ed|s)?\s+(?:in|by|for|as|to)?)\s*what\b",
            question.lower(),
        )
        if m:
            return m.group(1).strip()
        for rel in (
            "was founded in", "is located in", "was directed by",
            "published its findings in", "operated for", "was funded by",
        ):
            if rel in question.lower():
                return rel
        return ""

    @staticmethod
    def _answer_span(sentence: str, relation: str) -> str:
        """Extract the answer value from a sentence."""
        low = sentence.lower()

        if relation and relation in low:
            tail = sentence[low.index(relation) + len(relation) :]
            span = tail.strip().strip(".,;:").strip()
            if span and len(span) <= 45:
                return span

        m = re.search(r"(?:the answer is|answer:)\s*(.+?)(?:[.;]|$)", sentence, re.IGNORECASE)
        if m:
            span = m.group(1).strip().strip(".,;:")
            if span and len(span) <= 45:
                return span

        return ""
