"""Answer typing and adversarial target selection.

Why this module exists
----------------------

The attacker's target answer is not decoration. It is what decides whether the
*generation condition* of a poisoning attack can be satisfied at all.

The first version of this platform picked the target answer uniformly at random from
every answer in the dataset. That produced targets like "Nigel Lythgoe" for the question
"where does the optic nerve cross the midline". A poisoned passage asserting that is not
a plausible-but-wrong answer, it is a category error, and an instruction-tuned model
rejects it on sight. The measured consequence was stark: poison was retrieved on 100% of
runs but misled the generator only 43% of the time, against roughly 90% reported by
PoisonedRAG. The attack was never actually being tested.

The fix is to choose a target of the *same semantic type* as the gold answer: a year is
replaced by a different year, a person by a different person, a place by a different
place. This is what PoisonedRAG does with an LLM, reproduced here deterministically so
that corpora remain byte-reproducible from a seed.
"""

from __future__ import annotations

import random
import re

#: Coarse answer types. Deliberately shallow: the goal is to stop a person being
#: substituted for a year, not to build a named-entity recogniser.
ANSWER_TYPES = ("year", "date", "number", "person", "place_or_entity", "other")

_MONTHS = (
    "january february march april may june july august september october november "
    "december"
).split()

_YEAR = re.compile(r"^\s*(1[0-9]{3}|20[0-9]{2})\s*$")
_NUMERIC = re.compile(r"^\s*[\d,.]+\s*(%|percent|million|billion|thousand)?\s*$", re.I)
_HAS_DIGIT = re.compile(r"\d")

#: Tokens that mark a place or organisation rather than a person.
_PLACE_MARKERS = frozenset(
    """city state county province country river lake mountain island sea ocean bay
    university college school hospital company corporation institute ltd inc gmbh
    kingdom republic empire district region territory street avenue road park stadium
    airport station museum theatre theater church cathedral bridge""".split()
)

#: Common personal-name particles, used to separate people from other capitalised spans.
_PERSON_PARTICLES = frozenset("de la van von der den du el al bin ibn jr sr ii iii".split())


def answer_type(answer: str) -> str:
    """Classify an answer into one of :data:`ANSWER_TYPES`.

    Deterministic and dependency-free, so the corpus builder stays reproducible and
    runs with no model download.
    """
    text = (answer or "").strip()
    if not text:
        return "other"

    if _YEAR.match(text):
        return "year"

    low = text.lower()
    if any(month in low for month in _MONTHS) and _HAS_DIGIT.search(text):
        return "date"

    if _NUMERIC.match(text):
        return "number"

    tokens = text.split()
    if not _HAS_DIGIT.search(text) and 1 < len(tokens) <= 4:
        if any(t.lower() in _PLACE_MARKERS for t in tokens):
            return "place_or_entity"
        capitalised = [t for t in tokens if t[:1].isupper()]
        particles = [t for t in tokens if t.lower() in _PERSON_PARTICLES]
        # Two or three capitalised tokens with no place marker reads as a person name.
        if len(capitalised) + len(particles) >= 2 and len(tokens) <= 3:
            return "person"
        return "place_or_entity"

    if not _HAS_DIGIT.search(text) and len(tokens) == 1 and text[:1].isupper():
        return "place_or_entity"

    return "other"


class TargetAnswerPicker:
    """Chooses a plausible but incorrect answer for each query.

    The pool is grouped by :func:`answer_type`, and a target is drawn from the same
    group as the gold answer. When a group is too small to offer an alternative the
    picker falls back to the whole pool rather than failing, and records that it did so
    in :attr:`n_fallback` so the corpus validator can surface it.
    """

    #: A type group must hold at least this many distinct answers before it is trusted
    #: to supply a same-type target.
    MIN_GROUP = 4

    def __init__(self, pool: list[str], seed: int = 42) -> None:
        self.seed = seed
        self.n_fallback = 0
        self.n_typed = 0

        self._by_type: dict[str, list[str]] = {}
        for answer in pool:
            answer = (answer or "").strip()
            if answer:
                self._by_type.setdefault(answer_type(answer), []).append(answer)

        # Deduplicate while keeping order, so the choice is stable for a given seed.
        for key, values in self._by_type.items():
            self._by_type[key] = list(dict.fromkeys(values))

        self._all: list[str] = list(
            dict.fromkeys(a for group in self._by_type.values() for a in group)
        )

    def pick(self, gold: str, query_id: str = "") -> str:
        """Return a wrong answer of the same type as ``gold`` where possible."""
        gold = (gold or "").strip()
        # Seeding per query keeps the corpus reproducible and independent of iteration
        # order, which matters because cells are built lazily and cached.
        rng = random.Random(f"{self.seed}|target|{query_id}|{gold}")

        kind = answer_type(gold)
        group = [a for a in self._by_type.get(kind, []) if a.lower() != gold.lower()]

        if len(group) >= self.MIN_GROUP:
            self.n_typed += 1
            return rng.choice(group)

        wider = [a for a in self._all if a.lower() != gold.lower()]
        if not wider:
            self.n_fallback += 1
            return _perturb(gold, rng)

        self.n_fallback += 1
        return rng.choice(wider)

    def stats(self) -> dict[str, int]:
        return {
            "n_type_matched": self.n_typed,
            "n_fallback": self.n_fallback,
            "n_groups": len(self._by_type),
            **{f"group_{k}": len(v) for k, v in sorted(self._by_type.items())},
        }


def _perturb(gold: str, rng: random.Random) -> str:
    """Last-resort target when the pool offers no alternative at all.

    Produces something of the same shape as the gold answer rather than a random string,
    so the attack still has a chance of being believed.
    """
    kind = answer_type(gold)
    if kind == "year":
        try:
            return str(int(gold) + rng.choice([-7, -4, -3, 3, 4, 7]))
        except ValueError:
            pass
    if kind == "number":
        digits = re.sub(r"[^\d]", "", gold)
        if digits:
            return gold.replace(digits, str(int(digits) + rng.randint(1, 9)), 1)
    return f"{gold} (revised)"


def type_match_rate(pairs: list[tuple[str, str]]) -> float:
    """Share of (gold, target) pairs that share an answer type.

    Used by the corpus validator. A low rate means the attack is being handed targets
    the generator can dismiss without reasoning, which silently weakens every attack
    success rate the platform reports.
    """
    if not pairs:
        return 0.0
    same = sum(1 for gold, target in pairs if answer_type(gold) == answer_type(target))
    return same / len(pairs)
