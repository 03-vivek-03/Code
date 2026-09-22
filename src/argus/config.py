"""Configuration objects for the whole platform.

Everything is a plain dataclass so that configs are trivially serialisable into the
trace records, which matters because every result must be reproducible from its own
metadata alone.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- agent


@dataclass(frozen=True)
class AgentConfig:
    """One point in the mechanism ablation.

    The four mechanisms are the independent variables of Study A. There is deliberately
    no separate `stopping_policy` flag: a stopping decision cannot exist unless the agent
    can iterate, so it is a parameter of iterative retrieval and is varied through
    `iteration_budget` instead.

    On `max_context_docs`, which is not a cosmetic parameter
    --------------------------------------------------------

    It used to equal `top_k`, and that silently disabled M2 entirely. The answer prompt
    renders the best `max_context_docs` evidence items by score. Iterative retrieval
    excludes documents already seen, so every round after the first returns *strictly
    lower-scoring* documents. With `max_context_docs == top_k` those documents can never
    displace round one, and the generator sees an identical prompt no matter how many
    times the agent searched.

    The traces showed this exactly: at budgets 1, 2 and 3 the agent accumulated 4.98,
    9.86 and 14.55 evidence documents while input tokens stayed at 319.5, 319.5 and
    319.4, and attack success moved by 0.0005. The reported "iterative retrieval has no
    effect" was a disconnected instrument, not a null result.

    `max_context_docs` is therefore larger than `top_k` by default, and `__post_init__`
    refuses a configuration where iteration cannot reach the prompt.
    """

    name: str
    query_rewriting: bool = False
    iterative_retrieval: bool = False
    document_inspection: bool = False
    reflection: bool = False

    iteration_budget: int = 1
    top_k: int = 5
    max_rewrites: int = 2
    max_inspections: int = 2
    #: Documents rendered into the answer prompt. Must exceed `top_k` whenever the agent
    #: can iterate, or later rounds are collected and then discarded.
    max_context_docs: int = 10
    #: Whether an INSUFFICIENT reflection verdict appends the caution instruction that
    #: tells the model to abstain. Separating this from `reflection` is what lets the
    #: analysis distinguish "the agent reasoned past the poison" from "the agent stopped
    #: answering", which the measured data says are very different claims.
    reflection_caution: bool = False

    def __post_init__(self) -> None:
        if self.iteration_budget < 1:
            raise ValueError("iteration_budget must be at least 1")
        if not self.iterative_retrieval and self.iteration_budget != 1:
            raise ValueError(
                "iteration_budget > 1 requires iterative_retrieval=True; a stopping "
                "decision cannot exist without iteration"
            )
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.max_context_docs < self.top_k:
            raise ValueError(
                f"max_context_docs ({self.max_context_docs}) must be at least top_k "
                f"({self.top_k}), otherwise the first retrieval is itself truncated"
            )
        if self.iterative_retrieval and self.max_context_docs <= self.top_k:
            raise ValueError(
                f"config '{self.name}' enables iterative retrieval but "
                f"max_context_docs ({self.max_context_docs}) does not exceed top_k "
                f"({self.top_k}).\nLater rounds return strictly lower-scoring documents, "
                "so they could never enter the answer prompt and the mechanism would "
                "measure nothing. Raise max_context_docs above top_k."
            )
        if self.reflection_caution and not self.reflection:
            raise ValueError(
                "reflection_caution requires reflection=True; there is no verdict to "
                "act on otherwise"
            )

    @property
    def active_mechanisms(self) -> list[str]:
        out = []
        if self.query_rewriting:
            out.append("M1_rewrite")
        if self.iterative_retrieval:
            out.append("M2_iterate")
        if self.document_inspection:
            out.append("M3_inspect")
        if self.reflection:
            out.append("M4_reflect")
        return out

    @property
    def n_mechanisms(self) -> int:
        return len(self.active_mechanisms)

    def with_budget(self, budget: int) -> AgentConfig:
        """Return a copy with a different iteration budget (for the dose-response curve)."""
        return replace(self, name=f"{self.name}_b{budget}", iteration_budget=budget)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: The configurations of the ablation. C0 is the baseline, C1 to C4 enable exactly one
#: mechanism each, C5 enables all four.
#:
#: C6 is not part of the main ablation. It is the abstention control: reflection with
#: its caution instruction switched off, so the verdict is still computed and logged but
#: never changes the answer prompt. Comparing C4 against C6 separates the two things
#: reflection does — judging the evidence, and declining to answer — which the first run
#: conflated. That run reported reflection as a 12-point reduction in attack success;
#: the outcome breakdown showed refusals rising 13 points and correct answers *falling*
#: 2.9 points, so most of the "protection" was the agent going quiet.
MECHANISM_CONFIGS: dict[str, AgentConfig] = {
    "C0": AgentConfig(name="C0_vanilla"),
    "C1": AgentConfig(name="C1_rewrite", query_rewriting=True),
    "C2": AgentConfig(name="C2_iterate", iterative_retrieval=True, iteration_budget=3),
    "C3": AgentConfig(name="C3_inspect", document_inspection=True),
    "C4": AgentConfig(name="C4_reflect", reflection=True, reflection_caution=True),
    "C5": AgentConfig(
        name="C5_full",
        query_rewriting=True,
        iterative_retrieval=True,
        document_inspection=True,
        reflection=True,
        reflection_caution=True,
        iteration_budget=3,
    ),
    "C6": AgentConfig(
        name="C6_reflect_nocaution", reflection=True, reflection_caution=False
    ),
}

#: The configurations that make up the mechanism ablation proper. C6 is excluded because
#: it is a control on C4 rather than an independent mechanism.
ABLATION_CONFIGS = ["C0", "C1", "C2", "C3", "C4", "C5"]


# ----------------------------------------------------------------------- retrieval


@dataclass(frozen=True)
class RetrievalConfig:
    backend: str = "bm25"  # bm25 | dense | hybrid
    dense_model: str = "BAAI/bge-base-en-v1.5"
    top_k: int = 5
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    hybrid_alpha: float = 0.5  # weight on dense when backend == hybrid

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------- llm


@dataclass(frozen=True)
class LLMConfig:
    backend: str = "mock"  # mock | openai | local
    model: str = "mock-deterministic"
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 256
    timeout_s: float = 60.0
    max_retries: int = 3
    #: Attempts allowed once a call has been classified as failing for a transient reason
    #: (dropped connection, timeout, 5xx, rate limit) rather than a permanent one (bad key,
    #: malformed request). A run that died after ~3 seconds of total retry patience on a
    #: routine network blip is what this exists to prevent; see openai_compat.py.
    max_retries_transient: int = 8
    transient_backoff_cap_s: float = 60.0

    #: Only used by the mock backend. Controls how readily the simulated model is
    #: misled by poisoned context, before mechanism effects are applied.
    mock_susceptibility: float = 0.85

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["api_key"] = "***" if self.api_key else ""
        return d


# -------------------------------------------------------------------------- attack


@dataclass(frozen=True)
class AttackConfig:
    name: str = "poisonedrag_black"
    n_poison_docs: int = 5
    target_fraction: float = 1.0  # fraction of queries that are attack targets
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------- run


@dataclass(frozen=True)
class RunConfig:
    """A single cell of the experiment grid."""

    config_id: str = "C0"
    dataset: str = "synthetic"
    attack: str = "none"
    n_poison_docs: int = 0
    n_queries: int = 100
    retriever: str = "bm25"
    iteration_budget: int | None = None
    seed: int = 42

    @property
    def cell_id(self) -> str:
        budget = self.iteration_budget or 0
        return (
            f"{self.dataset}__{self.retriever}__{self.config_id}"
            f"__{self.attack}__p{self.n_poison_docs}__b{budget}"
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cell_id"] = self.cell_id
        return d


# -------------------------------------------------------------------------- global


@dataclass
class ArgusConfig:
    data_dir: Path = Path("data")
    seed: int = 42
    budget_usd: float = 35.0
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    #: Preflight every attack once per corpus and refuse to run one whose poison does
    #: not reach the top-k. Only tests, which use tiny corpora, should switch this off.
    check_attack_retrievability: bool = True
    #: Queries executed concurrently within a grid cell. The agent spends most of its
    #: wall-clock waiting on a remote generator, so this is close to a linear speed-up
    #: until the endpoint saturates. Trace files are byte-identical at any value.
    workers: int = 1

    # ---- derived paths -------------------------------------------------------
    @property
    def corpora_dir(self) -> Path:
        return self.data_dir / "corpora"

    @property
    def traces_dir(self) -> Path:
        return self.data_dir / "traces"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    def ensure_dirs(self) -> None:
        for p in (
            self.corpora_dir,
            self.traces_dir,
            self.results_dir,
            self.features_dir,
            self.models_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "seed": self.seed,
            "budget_usd": self.budget_usd,
            "llm": self.llm.to_dict(),
            "retrieval": self.retrieval.to_dict(),
        }


#: An unquoted .env value ends at the first '#' that follows whitespace.
_INLINE_COMMENT = re.compile(r"\s+#")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader so we do not take a dependency on python-dotenv.

    Handles the two things people actually write: `export KEY=value` and a trailing
    comment on the same line as a value. The comment case is not hypothetical — a value
    written as

        ARGUS_LLM_TIMEOUT_S=120     # 60s is tight for a full prompt

    previously became the literal string "120     # 60s is tight...", and the resulting
    `could not convert string to float` fired from inside `load_config`, so every command
    failed at once with an error naming neither the file nor the key.
    """
    if not path.exists():
        return

    # Read tolerantly. A single byte from a Windows-1252 paste — an em dash in a comment
    # is the usual culprit — used to raise UnicodeDecodeError out of `load_config`, which
    # every command calls, so the whole CLI died with an error that named neither the file
    # nor the character. A malformed comment must not be able to do that.
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
        print(
            f"  NOTE: {path} is not valid UTF-8; undecodable bytes were replaced. "
            "Check any non-ASCII characters in comments.",
            file=sys.stderr,
        )

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()

        # A quoted value is taken verbatim, so a key or URL containing '#' survives.
        # An unquoted value ends at the first whitespace-preceded '#', which is the usual
        # .env convention and leaves values like 'abc#def' alone.
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        else:
            val = _INLINE_COMMENT.split(val, 1)[0].strip()

        if key and key not in os.environ:
            os.environ[key] = val


def load_config(path: str | Path | None = None, **overrides: Any) -> ArgusConfig:
    """Build the global config from, in increasing order of precedence:

    1. package defaults
    2. an optional YAML file
    3. environment variables (including a .env file if present)
    4. keyword overrides
    """
    root = Path(__file__).resolve().parents[2]
    _load_dotenv(root / ".env")

    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    llm_raw = dict(data.get("llm", {}))
    ret_raw = dict(data.get("retrieval", {}))

    llm = LLMConfig(
        backend=_env("ARGUS_LLM_BACKEND") or llm_raw.get("backend", "mock"),
        model=_env("ARGUS_LLM_MODEL") or llm_raw.get("model", "mock-deterministic"),
        base_url=_env("ARGUS_LLM_BASE_URL") or llm_raw.get("base_url", ""),
        api_key=_env("ARGUS_LLM_API_KEY") or llm_raw.get("api_key", ""),
        temperature=float(llm_raw.get("temperature", 0.0)),
        max_tokens=int(_env("ARGUS_LLM_MAX_TOKENS") or llm_raw.get("max_tokens", 256)),
        # Reachable from .env because they matter on a remote endpoint: a 14B model
        # answering a full C5 prompt over HTTPS can take longer than the 60s default, and
        # a timeout there is indistinguishable from a failure.
        timeout_s=float(_env("ARGUS_LLM_TIMEOUT_S") or llm_raw.get("timeout_s", 60.0)),
        max_retries=int(_env("ARGUS_LLM_MAX_RETRIES") or llm_raw.get("max_retries", 3)),
        max_retries_transient=int(
            _env("ARGUS_LLM_MAX_RETRIES_TRANSIENT") or llm_raw.get("max_retries_transient", 8)
        ),
        transient_backoff_cap_s=float(
            _env("ARGUS_LLM_BACKOFF_CAP_S") or llm_raw.get("transient_backoff_cap_s", 60.0)
        ),
        mock_susceptibility=float(llm_raw.get("mock_susceptibility", 0.85)),
    )

    retrieval = RetrievalConfig(
        backend=_env("ARGUS_RETRIEVER") or ret_raw.get("backend", "bm25"),
        dense_model=_env("ARGUS_DENSE_MODEL") or ret_raw.get("dense_model", "BAAI/bge-base-en-v1.5"),
        top_k=int(ret_raw.get("top_k", 5)),
    )

    cfg = ArgusConfig(
        data_dir=Path(_env("ARGUS_DATA_DIR") or data.get("data_dir", "data")),
        seed=int(_env("ARGUS_SEED") or data.get("seed", 42)),
        budget_usd=float(_env("ARGUS_BUDGET_USD") or data.get("budget_usd", 35.0)),
        workers=int(_env("ARGUS_WORKERS") or data.get("workers", 1)),
        llm=llm,
        retrieval=retrieval,
    )

    for key, val in overrides.items():
        if val is not None and hasattr(cfg, key):
            setattr(cfg, key, val)

    return cfg


def get_agent_config(config_id: str, iteration_budget: int | None = None) -> AgentConfig:
    """Look up one of C0..C6, optionally overriding the iteration budget.

    A budget override also renames the configuration. That is not cosmetic: the runner
    caches the paired clean baseline under `clean__{dataset}__{retriever}__{name}`, and
    while the name stayed `C2_iterate` for every budget, all three budgets shared one
    cached clean-accuracy figure. Now that iteration genuinely changes the answer prompt,
    clean accuracy depends on the budget and must be measured per budget.
    """
    key = config_id.upper()
    if key not in MECHANISM_CONFIGS:
        raise KeyError(f"unknown config '{config_id}', expected one of {sorted(MECHANISM_CONFIGS)}")
    cfg = MECHANISM_CONFIGS[key]
    if iteration_budget is not None and iteration_budget != cfg.iteration_budget:
        if not cfg.iterative_retrieval:
            if iteration_budget != 1:
                raise ValueError(
                    f"config {key} has iterative_retrieval=False so its budget is fixed at 1"
                )
            return cfg
        return replace(
            cfg, iteration_budget=iteration_budget, name=f"{cfg.name}_b{iteration_budget}"
        )
    return cfg
