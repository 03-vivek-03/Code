"""Attack registry.

Named lookup so the CLI, the grid runner and the leave-one-attack-out evaluation all
refer to attacks by the same identifiers.
"""

from __future__ import annotations

from typing import Any

from argus.attacks.base import Attack
from argus.attacks.corpus_poisoning import CorpusPoisoningAttack
from argus.attacks.poisonedrag import PoisonedRAGBlackBox, PoisonedRAGWhiteBox

ATTACKS: dict[str, type[Attack]] = {
    "poisonedrag_black": PoisonedRAGBlackBox,
    "poisonedrag_white": PoisonedRAGWhiteBox,
    "corpus_poisoning": CorpusPoisoningAttack,
}

#: The three attacks used in the leave-one-attack-out protocol. Keeping this explicit
#: means the evaluation cannot silently drift if new attacks are added later.
LOAO_ATTACKS = ["poisonedrag_black", "poisonedrag_white", "corpus_poisoning"]


def list_attacks() -> list[str]:
    return sorted(ATTACKS)


def build_attack(name: str, n_poison_docs: int = 5, seed: int = 42, **kwargs: Any) -> Attack:
    key = name.lower()
    if key in ("none", ""):
        raise ValueError("'none' is not an attack; run the clean baseline instead")
    if key not in ATTACKS:
        raise KeyError(f"unknown attack '{name}', expected one of {list_attacks()}")
    return ATTACKS[key](n_poison_docs=n_poison_docs, seed=seed, **kwargs)
