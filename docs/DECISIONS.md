# Design decisions

Why the platform is built the way it is. Written for the thesis defence, where the
question is usually "why did you do it that way" rather than "what does it do".

---

## 1. Offline by default, with a mock LLM

**Decision.** The default backend is a deterministic simulated model. The entire
platform runs with no API key, no GPU and no network.

**Why.** Three reasons. Development iterations are free, so mistakes cost nothing. The
test suite runs on any machine, which matters because a suite that cannot run is not a
suite. And the pipeline can be demonstrated to a supervisor or a panel on a laptop with
no setup.

**The risk, and how it is managed.** Someone could mistake mock output for a result. So
the backend is recorded in every trace, the CLI prints a warning on every mock run, and
the dashboard shows a banner. Mock numbers exercise the pipeline; they are not findings.

**Why the mock is not a stub.** It parses the retrieved context, extracts the claims that
context makes, weighs them by position and emphasis, and answers accordingly. Poisoning
therefore has a real effect offline, and a test asserts the mock actually falls for
poison. A stub returning fixed strings would let a broken pipeline pass.

---

## 2. No stopping-policy mechanism

**Decision.** Four mechanisms, not five. The agent's stopping policy is not a separate
ablation configuration.

**Why.** A stopping decision cannot exist unless the agent can iterate. An early design
had a C5 for stopping policy, and it was inert: with a single retrieval there is nothing
to stop. It is a parameter of iterative retrieval, so it is varied through the iteration
budget instead, which also produces the dose-response curve for RQ3.

**Enforced in code.** `AgentConfig.__post_init__` raises if you set a budget above 1
without iteration enabled, so the mistake cannot be repeated.

---

## 3. Iterative retrieval excludes documents already seen

**Decision.** When M2 iterates without M1, the retriever excludes documents already
retrieved.

**Why.** Without exclusion, re-issuing the same query returns the same documents and
iteration does nothing. That was a real bug caught in early testing: C2 ran three
iterations and gathered five documents. Exclusion models an agent searching deeper into
the ranking, which is what an iterating agent with no query rewriter actually does.

**Tested.** `test_iteration_gathers_new_documents` asserts successive iterations return
disjoint document sets.

---

## 4. Reflection without iteration answers conservatively

**Decision.** M4 alone cannot search again on an INSUFFICIENT verdict, so it appends a
caution instruction to the answer prompt.

**Why.** Otherwise reflection would have no effect in isolation and C4 would be identical
to C0, making the ablation incomplete. Answering more carefully when evidence looks thin
is a real behaviour and gives reflection a distinct, measurable effect.

---

## 5. OpenTelemetry conventions instead of a new schema

**Decision.** Span attribute names follow the OpenTelemetry GenAI semantic conventions
and OpenInference. Only the `argus.*` namespace is project-specific.

**Why.** The conventions already cover retrieval, inference, tool execution and memory,
and production frameworks already emit them. Proposing another schema would be
non-novel, and it was explicitly ruled out during the literature review. Building the
security analytics layer above an existing standard is the contribution, and it means
the detector works against telemetry teams already collect.

---

## 6. Ground truth never touches a span

**Decision.** Poison document ids, gold answers and target answers live on the `Trace`
object. They are never span attributes.

**Why.** Features are computed from spans. Keeping ground truth off spans makes leakage
structurally impossible rather than merely avoided. A test also asserts that no feature
name contains a ground-truth word.

---

## 7. Leave-one-attack-out is the headline metric

**Decision.** The detector is evaluated by training on two attacks and testing on a
third it has never seen. The ordinary random split is computed but labelled
"optimistic upper bound".

**Why.** The obvious objection to Study B is that the detector merely memorised
self-generated attacks. Designing the evaluation around that objection from the start is
more honest, and more persuasive, than adding it during a rebuttal.

**Also mandatory.** False-positive rate at 95 percent detection, on clean traffic. A
detector that fires on normal queries is useless whatever its recall.

---

## 8. Baselines must declare their cost

**Decision.** `BaselineDefense.overhead_x` is part of the interface, expressed as a
multiple of one agent pass.

**Why.** The claim of Study B is not superior accuracy. It is comparable detection from a
different signal at near-zero cost. That claim only holds if the alternatives' costs are
measured rather than asserted. Putting it in the interface stops anyone reporting
accuracy and quietly omitting the price.

| Defence | Extra generator passes |
|---|---|
| Perplexity filter | 0 |
| LLM judge | k |
| LOO counterfactual | k+1 |
| Trace detector | 0 |

---

## 9. Attacks are reproductions, never new

**Decision.** Three published attacks, implemented from their papers. No novel attack.

**Why.** The contribution of the project is defensive. Publishing a new attack would
create disclosure obligations, complicate ethics approval, and add nothing to the
argument. Reusing published attacks with public code also makes the evaluation
comparable to prior work.

---

## 10. BM25 written directly rather than imported

**Decision.** Okapi BM25 implemented in about 100 lines of numpy instead of taking a
dependency.

**Why.** It removes a dependency, it runs anywhere including a free CPU instance, and it
is fully auditable. When a reviewer asks exactly how retrieval scores were computed, the
answer is in the repository rather than in someone else's package.

---

## 11. Traces are JSONL

**Decision.** Newline-delimited JSON, optionally gzipped. Not a database, not a
framework-specific format.

**Why.** The trace corpus is a released research artefact. Anyone should be able to read
it with the standard library, and the analysis has to survive any library churn in the
agent framework. Flushing after every write also makes a grid resumable, which matters on
a rate-limited free tier.

---

## 12. Statistics: intervals, effect sizes, FDR correction

**Decision.** Wilson intervals on every rate, Cohen's h for effect size, and
Benjamini-Hochberg correction across the mechanism comparisons.

**Why.** The claim of Study A is that a specific mechanism moves attack success by a
specific amount. A point estimate cannot support that. Wilson rather than the normal
approximation because attack success rates sit near 0 or 1, where the normal interval
gives impossible bounds. FDR correction because six simultaneous comparisons will produce
a spurious significant result if left uncorrected.

---

## 13. One full grid plus small confirmation runs

**Decision.** Full factorial on the primary axis (NQ, BM25, six configurations, three
attacks, three ratios), then small confirmation runs for multi-hop and dense retrieval.

**Why.** A full cross product of everything would multiply cost several times for very
little extra confidence. Confirmation runs are enough to show a finding is not an
artefact of one dataset or one retriever, and this structure is standard practice.

---

## 14. The clean baseline is paired

**Decision.** Clean accuracy is measured on the same queries with the same configuration
against the unpoisoned corpus, and cached per configuration **and iteration budget**.

**Why.** The utility cost of a mechanism and its security effect have to be measured on
identical ground, otherwise the comparison is confounded. Caching keeps the cost of the
pairing down, since the clean baseline does not depend on the attack.

**Revised.** The cache key originally omitted the iteration budget, because
`get_agent_config` did not rename a configuration when the budget was overridden. All
three budgets of C2 therefore shared one cached clean-accuracy figure. That was harmless
only while iteration could not reach the answer prompt; see revision R3 below.

---

# Revisions after the first full run

The first full programme produced 73,400 traces over 45 GPU-hours. An audit of those
results found eleven defects, five of them invalidating. The decisions below were changed
in response. They are recorded as revisions rather than edited into the originals,
because at a defence the useful question is not only "why is it built this way" but "what
did you get wrong and how do you know it is fixed".

Each revision names the measured evidence that exposed the problem, and the test that now
guards it.

---

## R1. A corpus must prove it is a retrieval corpus before it is used

**What went wrong.** The NQ loader never fetched passages. NQ-open ships only question and
answer strings, and rather than joining against a passage corpus the loader synthesised
one document per question reading `f"{question} The answer is {gold}."` BM25 matched the
question verbatim, clean accuracy read 0.98, and every downstream number described a
lookup task. The HotpotQA run — the only one using real paragraphs — was the control that
gave it away: accuracy there collapsed to 0.09 on the same pipeline.

**Decision.** Real datasets load real passages: NQ joined against the BEIR NQ passage
corpus, SQuAD as a light real-passage alternative, HotpotQA as before. Every corpus is
checked by `argus.corpus.validate` at build time and the builder **refuses to save** one
that fails. The synthetic generator remains, scoped to development, and is never
substituted silently.

**Why a hard failure.** A warning would have been ignored exactly as the silent version
was. The cost of a false refusal is minutes; the cost of a false pass was 45 hours.

**Guarded by.** `tests/test_regressions.py::TestD1CorpusIsARetrievalCorpus`.

---

## R2. Attack targets are type-matched

**What went wrong.** `_pick_target_answer` sampled uniformly from every answer in the
dataset, so the attacker's target for "where does the optic nerve cross the midline" was
"Nigel Lythgoe". PoisonedRAG's generation condition cannot be satisfied by a category
error. Measured: poison retrieved on 100% of runs, misleading the generator on 43%,
against roughly 90% in the paper.

**Decision.** `argus.corpus.answers` classifies answers into coarse types and draws the
target from the same type — a year for a year, a person for a person. Deterministic, so
corpora stay reproducible from a seed, and no model download.

**Also.** The attack templates no longer splice question fragments into a sentence as the
grammatical subject, which is what produced "where does the optic is Nigel Lythgoe". They
assert the answer in sentences that are grammatical for any question, at a passage length
comparable to real prose.

**Guarded by.** `TestD2TargetAnswersArePlausible`, plus the `target_type_match` check in
the corpus validator.

---

## R3. `max_context_docs` must exceed `top_k` when the agent can iterate

**What went wrong.** Both were 5. The answer prompt renders the best `max_context_docs`
evidence items by score, and iterative retrieval excludes what it has already seen, so
every round after the first returns strictly lower-scoring documents that can never
displace round one. At budgets 1, 2 and 3 the agent gathered 4.98, 9.86 and 14.55
documents while input tokens stayed at 319.5, 319.5 and 319.4 and attack success moved by
0.0005.

The published conclusion — "iterative retrieval has no effect, Cohen's h = 0.006" — was a
disconnected wire, not a null result. RQ3 was unmeasured rather than answered.

**Decision.** `max_context_docs` defaults to 10 against `top_k` 5, and
`AgentConfig.__post_init__` raises on any configuration where iteration cannot reach the
prompt.

**Guarded by.** `TestD4IterationReachesThePrompt`, which asserts the rendered prompt
actually grows with the budget rather than merely that iteration happened.

---

## R4. The detection label is the compromise event, not the attack's outcome

**What went wrong.** `label = "compromised" if (poison_ids and attack_success) else
"benign"`. A run that was attacked, retrieved poison into 80% of its prompt, and happened
to answer correctly was filed as benign — behaviourally identical to the compromised run
beside it. 26,694 traces (39.3% of all attacked runs) sat in the benign class with poison
in their prompt, against 5,200 genuinely clean traces, so 83.7% of the negative class was
poisoned.

The detector was therefore being asked to predict whether a language model would be
fooled, from trajectory features that cannot carry that information. Leave-one-attack-out
recall was 0.052, F1 0.091, and the strongest feature was `answer_is_refusal` at 0.354
importance — the model reading its own outcome off the answer string.

**Decision.** Two labels, carried separately. `label` is the compromise **event**:
attacked, and poison reached the answer prompt. `attack_success` is the **outcome** and
remains Study A's dependent variable. `label_reason` records which rule applied, so the
released trace corpus is self-documenting.

**Why the event is the right target.** A defender can act on it. A run where poison
reached the context is compromised whether or not the generator happened to survive it,
and that is the thing external telemetry can plausibly reveal.

**Guarded by.** `TestD5LabelIsTheCompromiseEvent` and two integration tests that assert no
negative has poison in its prompt.

---

## R5. Reflection's two effects are separated

**What went wrong.** Reflection was reported as a 12-point reduction in attack success.
The outcome breakdown showed refusals rising 13.0 points and correct answers *falling*
2.9, with clean accuracy dropping from 0.980 to 0.948. Split by verdict: where the caution
instruction fired, correctness collapsed from 0.808 to 0.072 and the attack still landed
28.0% of the time, against 15.8% in the confident branch.

The mechanism was abstention, not resistance. Reported as a single attack-success figure,
an agent that stops answering is indistinguishable from one that reasons past poison.

**Decision.** `reflection_caution` is a separate switch from `reflection`, and
configuration **C6** is reflection with it disabled — the verdict is still computed and
logged, but never changes the answer prompt. C6 runs across the whole main grid alongside
C4, so the decomposition is available at every attack and ratio.

Abstention is now a first-class outcome throughout: `Trace.is_abstention`,
`RunOutcome.abstention_rate`, `asr_when_answered`, and `abstention_table.csv` with a
`share_explained_by_abstention` column.

**Why this is a better result, not a smaller one.** "The agentic safety margin reported in
the literature is substantially an abstention effect, and it is not free" is a sharper
contribution than confirming someone else's hypothesis.

**Guarded by.** `TestD7AbstentionIsMeasured` and a test asserting C4 and C6 differ in
exactly one field.

---

## R6. Context metrics describe the prompt, ranks are global

**What went wrong.** `context_doc_ids` unioned every retrieval span instead of reading
what was rendered, so C2's poisoned context fraction read 0.343 when the prompt was 0.447.
`poison_rank` took the minimum rank within each span, and because later rounds exclude
what was already seen and renumber from zero, a document at true rank 5 reported rank 0 —
which is the entire reason iterative retrieval appeared to rank poison *better* (0.28)
than vanilla (0.50).

**Decision.** The answer span records `argus.context_document_ids`, the exact set sent to
the generator, and every "context" metric reads it. Retrieval spans record
`retrieval.global_ranks`, the position in the unfiltered ranking, and `poison_rank` uses
those. `poison_retrieved` is kept as a separate, honestly named superset.

**Guarded by.** `TestD8ContextMetricsDescribeThePrompt`.

---

## R7. Leave-one-attack-out holds the attack out of training

**What went wrong.** The benign pool was `attack in {none, ""} or y == 0`, so every
*unsuccessful* run of the held-out attack counted as benign and was split 70/30 into train
and test. The detector saw the held-out attack's behavioural distribution during training
— precisely what the protocol exists to prevent.

Separately, the `corpus_poisoning` fold held 46 positives in 16,459 rows because that
attack never retrieved, and pooling it dragged every detector below chance.

**Decision.** Negatives are partitioned by origin: genuinely clean traces are shared
across folds, attacked-but-negative traces follow their own attack into train or test. A
fold with fewer than 50 positives is reported as skipped, with the reason, rather than
pooled.

**Guarded by.** `TestLOAOHasNoLeakage`, which inspects the actual training matrix.

---

## R8. Cost is reported, never a design constraint

**What went wrong.** `inference_overhead_x` was hardcoded to 0.0 at four call sites, so
the cost column read as "not measured" rather than "measured, and zero". Worse,
`argus baselines compare` required a poisoned corpus on disk and nothing in the pipeline
ever wrote one, so it failed every time it was run and `baseline_comparison.csv` was never
produced. The central claim of Study B — comparable detection at a fraction of the
inference cost — shipped with no supporting evidence at all.

**Decision.** `TRACE_DETECTOR_OVERHEAD_X` is a named constant used everywhere. The
baseline comparison reconstructs the poisoned corpus from the traces themselves and
includes clean traces so both classes are present. The grid's spend ceiling warns rather
than refuses, and `ARGUS_BUDGET_USD=0` disables the mid-run guard: on a local endpoint the
dollar figure is notional, and a guard that silently truncates a 34-hour grid costs more
than it saves.

**Guarded by.** `TestD9CostIsMeasuredNotPlaceheld`.

---

## R9. Resume only from a cell that is actually complete

**What went wrong.** One cell's trace file held 141 usable records — the 141st truncated
mid-string — while its result JSON recorded `n_traces: 500`. Study A counted 500 runs and
Study B saw 140. The runner skipped the cell on resume because the file existed.

**Decision.** `count_valid_traces` counts parseable records and corrupt lines, and the
runner re-runs any cell that does not hold `n_queries` valid records. `TraceReader` skips
corrupt lines and counts them rather than raising, so analysis reports damage instead of
refusing to start. `argus validate data` cross-checks every result file against its trace
file.

**Guarded by.** `TestD10IntegrityAndReachableThresholds`.

---

## R10. Grids must compare configurations on the same queries

**What went wrong.** C0, C1 and C2 ran 1,000 queries; C3, C4 and C5 ran 500. Cells answer
`queries[:n_queries]`, so the comparison crossed different query sets. C0's attack success
rate is 0.3029 over the full thousand and 0.2873 over the matched prefix, so C4's headline
reduction was reported as −12.0 points rather than its true −9.6, and C5's as −11.4 rather
than −9.6.

**Decision.** `check_matched_queries` refuses a grid whose cells within a dataset disagree
on `n_queries`, and `build_grid` calls it. The analysis restricts to the intersection of
query sets by default, with `--no-matched` to opt out.

**Guarded by.** `TestD6MatchedQuerySets`, including a check that every shipped preset is
internally matched.

---

## R11. The rule detector has a reachable operating point

**What went wrong.** It scored the fraction of six 90th-percentile rules that fired, and
evaluation thresholded that at 0.5 — requiring three independent tail events on one trace.
Recall was 0.000 in every fold and pooled AUC 0.408, below chance. The "simple baseline"
row of the comparison was meaningless.

**Decision.** The score is continuous, each rule contributing how far the trace sits into
the suspicious tail of the benign distribution, and the operating point is calibrated at
fit time so 0.5 corresponds to a chosen false-positive rate on benign traffic.

**Guarded by.** `TestD10IntegrityAndReachableThresholds::test_rules_detector_can_actually_fire`.

---

## R12. The mock reads documents, not sentences

**What went wrong.** Not a defect of the first run, but exposed by fixing R2. The mock
judged relevance sentence by sentence, so a passage that introduced its subject in one
sentence and asserted the fact in another could not poison it. That held only for one
specific template shape; when the attack templates were made grammatical, the mock stopped
being misled and the offline poisoning test failed.

**Decision.** Relevance is judged per document block, and the answer span is taken from
whichever sentence inside a relevant block carries one. That is closer to how a real model
reads, and it makes the offline test sensitive to attack changes rather than to one
template.

**Guarded by.** `tests/test_agent.py::TestDeterminism::test_mock_responds_to_poisoned_context`.
