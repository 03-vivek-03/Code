"""Knowledge-poisoning attacks.

Every attack here is a reproduction of published work. No new attack is proposed. They
exist so that defences can be evaluated against a realistic adversary, which is the
standard practice in this literature and the reason the contribution of this project is
defensive.

Implemented:

* ``poisonedrag_black``  PoisonedRAG in its black-box setting (Zou et al., USENIX
  Security 2025). The attacker knows the query but not the retriever, so the poisoned
  passage is crafted to be lexically close to the query.
* ``poisonedrag_white``  PoisonedRAG in its white-box setting. The attacker also knows
  the retriever and optimises the passage against it directly.
* ``corpus_poisoning``   Untargeted adversarial passages (Zhong et al., EMNLP 2023).
  One passage aims to be retrieved across many unrelated queries.
"""

from argus.attacks.base import Attack, AttackResult
from argus.attacks.corpus_poisoning import CorpusPoisoningAttack
from argus.attacks.poisonedrag import PoisonedRAGBlackBox, PoisonedRAGWhiteBox
from argus.attacks.registry import ATTACKS, build_attack, list_attacks

__all__ = [
    "ATTACKS",
    "Attack",
    "AttackResult",
    "CorpusPoisoningAttack",
    "PoisonedRAGBlackBox",
    "PoisonedRAGWhiteBox",
    "build_attack",
    "list_attacks",
]
