"""Parsing model responses into the values the engine acts on.

Why this is its own module
--------------------------

Three points in the agent loop turn free text from the generator into a control decision:
which document to inspect, whether the evidence is sufficient, and what to search for
next. Each was originally a one-liner inside the engine, tuned against one model on one
serving stack, and each failed *silently* on anything that answered differently:

* **inspect** collected every digit in the response and concatenated them, so
  "Document 3 is best because it covers 2 topics" produced index 32, which was then
  clamped to the last candidate. The agent inspected the wrong document and nothing said so.
* **reflect** tested ``"INSUFFICIENT" in text.upper()`` and defaulted to SUFFICIENT
  otherwise, so a model replying "Not sufficient" disabled the mechanism entirely while
  appearing to work.
* **rewrite** used the whole response as the next search query, so a preamble like
  "Here is a search query:" went into the retriever as query terms.

Moving to a different model or a different endpoint is exactly when these break, and the
whole point of this project is that a silently degraded mechanism produces numbers that
look fine. So the parsers live here, they are tested directly, and every one of them
reports whether it understood the response or fell back. The engine records that flag on
the span, and the runner surfaces the rate per cell.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

#: First run of digits anywhere in the text. Deliberately not "all digits": a chatty
#: response mentions numbers that are not the answer.
_FIRST_INT = re.compile(r"\d+")

#: Words that mark the text before a colon as scaffolding rather than content, e.g.
#: "Here is a new search query: ..." or "Query: ...".
#:
#: Enumerating English preambles with one regex was tried and was too brittle — it missed
#: "Here is a new search query:" because of the intervening "a". Testing what sits before
#: the colon is both simpler and more precise, and it leaves a legitimate query containing
#: a colon ("optic chiasm: anatomy") untouched, because none of these words appear in it.
_PREAMBLE_MARKERS = frozenset(
    "query queries search here answer suggest recommend rewrite reformulated "
    "revised alternative following".split()
)

#: How far into the response a colon can be and still plausibly end a preamble.
_MAX_PREAMBLE_CHARS = 70

_WORD = re.compile(r"[a-z']+")

#: Phrasings that mean the evidence is NOT enough. Checked before the positive form,
#: because "not sufficient" contains "sufficient".
_INSUFFICIENT_PATTERNS = (
    r"\binsufficient\b",
    r"\bnot\s+(?:enough|sufficient)\b",
    r"\bno\b.{0,20}\bsufficient\b",
    r"\bcannot\s+(?:be\s+)?(?:determined?|answered?)\b",
    r"\bmore\s+(?:evidence|information|context)\s+(?:is\s+)?(?:needed|required)\b",
    r"\bincomplete\b",
)
_SUFFICIENT_PATTERNS = (
    r"\bsufficient\b",
    r"\benough\b",
    r"\badequate\b",
    r"\byes\b",
)

_INSUFFICIENT_RE = re.compile("|".join(_INSUFFICIENT_PATTERNS), re.IGNORECASE)
_SUFFICIENT_RE = re.compile("|".join(_SUFFICIENT_PATTERNS), re.IGNORECASE)


@dataclass
class Parsed(Generic[T]):
    """A parsed value plus whether the response was actually understood.

    ``ok`` is False when the parser could not find what it needed and used a fallback.
    That distinction is the whole reason this type exists: a fallback is a legitimate way
    to keep a run going, but a run where half the reflect calls fell back is not measuring
    reflection, and the analysis has to be able to tell.
    """

    value: T
    ok: bool
    raw: str = ""
    reason: str = ""


def parse_document_choice(text: str, n_candidates: int) -> Parsed[int]:
    """Read a zero-based document index out of an inspect response.

    The prompt asks for a document number, one-based. Takes the **first** integer in the
    response and clamps it into range. A response with no integer at all falls back to the
    top-ranked candidate, which is the sensible default, but is reported as a failure.
    """
    if n_candidates <= 0:
        return Parsed(0, False, text, "no candidates")

    match = _FIRST_INT.search(text or "")
    if not match:
        return Parsed(0, False, text, "no integer in response")

    one_based = int(match.group())
    idx = one_based - 1
    if idx < 0 or idx >= n_candidates:
        return Parsed(
            max(0, min(idx, n_candidates - 1)),
            False,
            text,
            f"index {one_based} outside 1..{n_candidates}",
        )
    return Parsed(idx, True, text)


def parse_reflection(text: str) -> Parsed[str]:
    """Read a SUFFICIENT / INSUFFICIENT verdict out of a reflect response.

    Negative forms are matched first, because "not sufficient" contains "sufficient" and
    the original substring test therefore read it as approval.

    When neither form is present the verdict falls back to SUFFICIENT — the conservative
    choice, since it leaves the answer prompt untouched — but ``ok`` is False so the run
    is counted as a parse failure rather than treated as a real verdict.
    """
    blob = (text or "").strip()
    if not blob:
        return Parsed("SUFFICIENT", False, text, "empty response")

    if _INSUFFICIENT_RE.search(blob):
        return Parsed("INSUFFICIENT", True, text)
    if _SUFFICIENT_RE.search(blob):
        return Parsed("SUFFICIENT", True, text)
    return Parsed("SUFFICIENT", False, text, "no verdict found")


def _strip_preamble(blob: str) -> str:
    """Drop a leading "Here is a search query:"-style prefix.

    Only fires when the text before the colon actually reads as scaffolding, so a query
    that legitimately contains a colon survives intact.
    """
    idx = blob.rfind(":", 0, _MAX_PREAMBLE_CHARS)
    if idx <= 0:
        return blob.strip()

    head = blob[:idx].lower()
    if not _PREAMBLE_MARKERS & set(_WORD.findall(head)):
        return blob.strip()

    tail = blob[idx + 1 :].strip()
    # "Here is a search query:" with nothing after it is a failure, not a rewrite; return
    # the empty string so the caller falls back rather than searching for the preamble.
    return tail


def parse_rewrite(text: str, original: str, max_ratio: float = 3.0) -> Parsed[str]:
    """Read a search query out of a rewrite response.

    Strips a leading preamble and surrounding quotes, then rejects results that are empty
    or implausibly long. A rejected rewrite falls back to the original question, which
    makes the mechanism a no-op for that run rather than poisoning retrieval with prose.
    """
    blob = (text or "").strip()
    if not blob:
        return Parsed(original, False, text, "empty response")

    # A model that explains itself usually puts the query on its own line first.
    first_line = blob.splitlines()[0].strip()
    if len(first_line) >= 8:
        blob = first_line

    stripped = _strip_preamble(blob)
    stripped = stripped.strip("\"'`").strip()
    # A trailing full stop is prose punctuation, not a query term.
    stripped = stripped.rstrip(".").strip()

    if not stripped:
        return Parsed(original, False, text, "nothing left after stripping preamble")

    limit = max(len(original) * max_ratio, 80)
    if len(stripped) > limit:
        return Parsed(original, False, text, f"rewrite {len(stripped)} chars exceeds {limit:.0f}")

    return Parsed(stripped, True, text)


def parse_answer(text: str) -> Parsed[str]:
    """Normalise an answer response.

    The answer prompt asks for the value alone. Nothing is rejected here — an over-long
    answer is a real behaviour worth measuring, and `Trace._match` is a substring test so
    verbosity does not break scoring. The parser only trims whitespace and reports whether
    the response was empty.
    """
    blob = (text or "").strip()
    return Parsed(blob, bool(blob), text, "" if blob else "empty response")
