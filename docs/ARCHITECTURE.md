# Architecture

How the platform fits together, and why it is built this way.

---

## The one-sentence version

One instrumented agentic RAG system produces execution traces. Study A analyses those
traces to find which agentic mechanism changes poisoning susceptibility and at which
stage. Study B trains a detector on the same traces. Study B collects no data of its own.

---

## Data flow

```
corpus (NQ, HotpotQA, synthetic)
    |
    +-- attack injects poisoned documents        attacks/
    |
    v
poisoned corpus
    |
    +-- retriever indexes it                     retrieval/
    |
    v
agentic RAG engine                               agent/
    [M1 rewrite] [M2 iterate] [M3 inspect] [M4 reflect]
    |
    +-- every action emits an OpenTelemetry span telemetry/
    |
    v
labelled execution traces (JSONL)
    |
    +----------------------------+
    |                            |
    v                            v
STUDY A                      STUDY B
analysis/                    features/ -> detect/
ablation, stage split        detector, LOAO evaluation
"which mechanism, and       "can we catch it at runtime?"
 at which stage?"
```

---

## Modules

| Module | Responsibility | Key design decision |
|---|---|---|
| `config` | Dataclass configuration for everything | Frozen dataclasses so a config serialises straight into the trace; every result is reproducible from its own metadata |
| `corpus` | Document store, dataset loaders, synthetic generator | The synthetic generator exists so the whole platform runs offline; it mirrors NQ and HotpotQA structure so the code path is the same |
| `retrieval` | BM25, dense, hybrid | BM25 is written directly in numpy. No heavy dependency, runs anywhere, and is fully auditable for the thesis |
| `llm` | Mock, OpenAI-compatible, local HF | The mock reads context and answers from the balance of evidence, so poisoning has a real effect offline |
| `agent` | The four mechanisms and the execution loop | Each mechanism is independently switchable. This is what makes the ablation an ablation |
| `attacks` | PoisonedRAG black and white box, corpus poisoning | Reproductions of published work. No new attack is proposed |
| `telemetry` | Spans, tracer, JSONL writer | Attribute names follow the OpenTelemetry GenAI conventions rather than a bespoke schema |
| `features` | Six feature families | Nothing may read model internals or re-invoke the generator. Enforced by a test |
| `detect` | Rules, Isolation Forest, GBDT, evaluation | Leave-one-attack-out is the headline protocol, built in from the start |
| `analysis` | Ablation, stage decomposition, statistics | Effect sizes with confidence intervals and FDR correction, not bare point estimates |
| `baselines` | Perplexity filter, LLM judge, LOO counterfactual | Each declares its inference overhead, because the cost comparison is the argument |
| `runner` | Grid execution, cost control, resumability | Traces flush after every run; a grid that dies keeps its completed work |

---

## The four mechanisms

| Config | M1 rewrite | M2 iterate | M3 inspect | M4 reflect |
|---|---|---|---|---|
| C0 vanilla | no | no | no | no |
| C1 | yes | no | no | no |
| C2 | no | yes | no | no |
| C3 | no | no | yes | no |
| C4 | no | no | no | yes |
| C5 full agentic | yes | yes | yes | yes |

**Why there is no stopping-policy configuration.** A stopping decision cannot exist
unless the agent can iterate, so it is not independent. It is handled as a parameter of
M2 through the iteration budget, varied across 1, 2 and 3, which also produces the
dose-response curve for RQ3. `AgentConfig` raises if you try to set a budget above 1
without iteration enabled.

**Two subtleties in isolating mechanisms.**

*M2 without M1* re-issues the same query, so the retriever excludes documents already
seen. That models an agent searching deeper into the ranking rather than uselessly
repeating round one. A test asserts successive iterations return disjoint document sets.

*M4 without M2* cannot act on an INSUFFICIENT verdict by searching again, because there
is no iteration. It acts by answering more conservatively. That is a real behaviour and
gives reflection a distinct, measurable effect on its own.

---

## Telemetry

Attribute names follow the OpenTelemetry GenAI semantic conventions:

| Span kind | Convention name | Carries |
|---|---|---|
| Retrieval | `retrieval.query` | query, `retrieval.document_ids`, `retrieval.scores`, iteration |
| Inference | `gen_ai.chat` | `gen_ai.request.model`, `gen_ai.usage.input_tokens`, task |
| Tool | `gen_ai.execute_tool` | `gen_ai.tool.name`, arguments, result preview |
| Mechanism | `argus.mechanism` | which mechanism fired and what it decided |

Only the `argus.*` namespace is project-specific, and it carries the mechanism decision
points that make the ablation legible. Ground truth (poison ids, gold answers) lives on
the `Trace`, never on a span, so it can never leak into a feature.

This is deliberate. OpenTelemetry already standardises this telemetry and production
frameworks already emit it, so inventing another schema would contribute nothing. The
contribution is the security analytics layer above an existing standard.

---

## The stage decomposition

```
P(attack succeeds) = P(poison enters context) x P(model misled | poison in context)
                          retrieval stage              reasoning stage
```

Both terms come from data the ablation already collects, so this costs no extra
experiments. It works only because the platform logs which documents actually entered
context, which is exactly the response-level logging prior work identified as missing.

`analysis/stage.py` also reports `asr_without_poison_in_context`. If the model produces
the attacker's answer without ever seeing poison, the decomposition is contaminated, so
that is surfaced rather than hidden.

---

## Detection

Three detectors, deliberately ordered simple to complex so the value of complexity is
demonstrated rather than assumed:

1. **rules** interpretable thresholds. The floor. If this works, trace features carry an
   obvious signal, which is itself a finding.
2. **iforest** trained on benign traces only. The realistic deployment case, since a
   defender rarely has labelled attacks.
3. **gbdt** supervised. The ceiling.

**Evaluation is designed around the objection that would otherwise sink the work:** that
the detector merely memorised the attacks it was trained on.

| Protocol | What it answers |
|---|---|
| `leave_one_attack_out` | Does it work on an attack it has never seen? **The headline number** |
| `fpr_at_95_tpr` | How often does it fire on clean traffic? |
| `cross_dataset` | Does it transfer between datasets? |
| `feature_ablation` | Is any single feature family carrying everything? |
| `random_split` | Upper bound only, labelled optimistic |

---

## Extension points

| To add | Do this |
|---|---|
| An attack | Subclass `Attack`, implement `craft()`, register in `attacks/registry.py` |
| A retriever | Subclass `Retriever`, implement `build()` and `_search()`, add to the factory |
| An LLM backend | Subclass `LLMBackend`, implement `_generate()`, add to the factory |
| A feature | Add it to a family in `features/schema.py` and compute it in `extractor.py` |
| A detector | Subclass `Detector`, implement `fit()` and `score()`, register in `detect/models.py` |
| A baseline defence | Subclass `BaselineDefense`, set `overhead_x` honestly |

Adding an attack automatically extends the leave-one-attack-out evaluation, which is why
`LOAO_ATTACKS` is listed explicitly in the registry rather than inferred.
