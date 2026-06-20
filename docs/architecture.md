# Architecture

How the pieces fit, why they fit that way, and where the honesty is *enforced* rather
than asserted. For the product pitch and the leaderboard, see the
[root README](../README.md); for the meta-eval methodology, see [`eval.md`](eval.md).

> **Legend (used in every diagram below):**
> 🟦 deterministic (pure code — same input, same output) ·
> 🟨 LLM (distributional — N runs, never byte-stable) ·
> 🟥 honesty rail (degradation / N-A short-circuit — never a crash, never a wrong verdict).

---

## The load-bearing principle

**`core` is a plain importable Python module. Everything that *can* import it does so
directly — there is no service-to-service HTTP between Python components.** The CLI, the
FastAPI adapter, and the eval harness all call `from groundcheck import check` in-process.
The only component that talks HTTP is the browser, because a browser cannot import a Python
module.

That single rule is what makes the eval harness parallel-safe and the cost honest: there is
no network hop to mock, no service to stand up, and `check()` is a pure function of
`(source, answer)` plus a provider seam.

```mermaid
flowchart LR
    subgraph importers["import core directly (in-process, no HTTP)"]
        CLI["core/ cli<br/>python -m groundcheck.cli"]
        EVAL["eval/ harness<br/>two-tier meta-eval"]
    end
    API["api/: thin FastAPI adapter<br/>(reshape + error-map only,<br/>no engine logic)"]
    CORE["core/: the engine<br/>decompose · ground ×N · score · highlight<br/>+ provider seam + metrics"]
    UI["app/: GroundCheck.dc.html<br/>(browser; renders only)"]
    PROV["provider seam<br/>Anthropic · Azure OpenAI · Mock"]

    CLI --> CORE
    EVAL --> CORE
    API --> CORE
    CORE --> PROV
    UI -- "HTTP / fetch('/check')" --> API

    style CORE fill:#dbeafe,stroke:#1e40af
    style CLI fill:#dbeafe,stroke:#1e40af
    style EVAL fill:#dbeafe,stroke:#1e40af
    style API fill:#dbeafe,stroke:#1e40af
    style PROV fill:#fef9c3,stroke:#a16207
    style UI fill:#ede9fe,stroke:#6d28d9
```

- **`core`** is the orchestrator (`pipeline.check`). `import groundcheck` pulls in only the
  data contracts and prompts — **never** `anthropic`/`openai` (both are lazy-imported inside
  the providers), so the package imports **key-free and fast**.
- **`api`** is a *thin* adapter: validate the request, call `check()`, capture engine
  warnings, map every error to clean JSON. It mounts the static `app/` **same-origin** under
  `/app`, so the page `fetch`es `/check` with no CORS. The "no grounding/scoring/highlighting
  logic in `api/`" rule is real discipline — `api/` never touches a verdict.
- **`app`** is the demo UI (no build step). It marks one span per claim and surfaces every
  edge state (loading, error, missing-key, N/A, refusal, borderline, unlocated).
- **`eval`** imports `core` directly, delegates **all** metric math to `groundcheck.metrics`,
  and is the only layer that ships the gold datasets (`pyyaml` is an eval-only dependency, so
  the engine stays dataset-free).

---

## The check pipeline

`check(source, answer)` is the whole product. It decomposes the answer into atomic claims,
grounds each claim **against the source only** N times, resolves a majority verdict with a
*flagging* bias, scores `SUPPORTED / n_claims`, and highlights the answer. The LLM steps are
distributional; everything around them is deterministic code — including every honesty rail.

```mermaid
flowchart TD
    A([Answer]) --> CAPA["cap answer to token budget<br/>warn + truncate, never balloon cost"]
    S([Source]) --> CAPS["cap source + cache-floor gate<br/>code, no LLM"]

    CAPA --> DEC["DECOMPOSE<br/>Sonnet 4.6, structured output"]
    DEC --> Z{"0 claims?<br/>code"}
    Z -- "yes (empty / whitespace)" --> NA["faithfulness_score = None<br/>shown as N-A, never a fake 100%"]

    Z -- "no" --> GR["GROUND each claim ×N<br/>Opus 4.8, concurrent (6-worker pool)<br/>SOURCE sent as cached block"]
    CAPS --> GR
    GR --> REF{"run refused?<br/>safety classifier"}
    REF -- "yes" --> NEI["map run → NOT_ENOUGH_INFO<br/>n_refused surfaced, not hidden"]
    REF -- "no" --> VOTE
    NEI --> VOTE["majority label + SEVERITY tie-break<br/>CONTRADICTED &gt; NEI &gt; SUPPORTED<br/>split vote biases to FLAG (code)"]

    VOTE --> SCORE["score = n_supported / n_claims<br/>+ counts + confidence (code)"]
    SCORE --> HL["highlight answer<br/>locate ladder: exact → regex → prefix → give up<br/>unlocated = shown, not highlighted (code)"]
    HL --> OUT([cited report:<br/>per-claim verdicts · votes · score · green/amber/red])

    style CAPA fill:#dbeafe,stroke:#1e40af
    style CAPS fill:#dbeafe,stroke:#1e40af
    style Z fill:#dbeafe,stroke:#1e40af
    style VOTE fill:#dbeafe,stroke:#1e40af
    style SCORE fill:#dbeafe,stroke:#1e40af
    style HL fill:#dbeafe,stroke:#1e40af
    style DEC fill:#fef9c3,stroke:#a16207
    style GR fill:#fef9c3,stroke:#a16207
    style NA fill:#fee2e2,stroke:#b91c1c
    style NEI fill:#fee2e2,stroke:#b91c1c
```

Four decisions in that diagram carry the design:

1. **0 claims → `None`, not 100%.** An empty or whitespace answer (or one the model cannot
   decompose) short-circuits *before* any grounding call, with `cost_usd == 0.0`. The score
   is `None`, rendered as **N/A** — never a misleading "100% faithful" for an answer that
   said nothing.
2. **Grounding is fanned out, assembled in document order.** Claims ground concurrently in a
   6-worker pool, but results are slotted by claim index, so the report's order never depends
   on completion order. A single claim's grounding failure is logged and degraded to NEI — it
   never aborts the check.
3. **The tie-break biases toward flagging.** The majority vote is **not**
   `Counter.most_common` (arbitrary on ties). Among labels tied for the top count it returns
   the **most severe** present (`CONTRADICTED > NOT_ENOUGH_INFO > SUPPORTED`). A split vote
   flags; it never silently certifies a claim as grounded. That is the correct bias for a
   firewall.
4. **Highlighting is a soft link.** Locating a claim's sentence in the original answer is
   best-effort (exact → whitespace-tolerant regex → ~40-char prefix → give up). A sentence
   that can't be located is listed as *unlocated* and shown **unhighlighted** — it degrades
   to "claim shown, not highlighted," never to a wrong verdict.

### N-run majority — the determinism engine

Opus 4.8 rejects `temperature`/`top_p`/`top_k` and has no `seed`, so a single grounding call
is a coin flip on borderline claims. Determinism is *engineered*: each claim is grounded N
times (default 3) and resolved by majority with the severity tie-break.

```mermaid
flowchart LR
    C([one claim + source]) --> R1["ground_once #1<br/>Opus 4.8"]
    C --> R2["ground_once #2<br/>Opus 4.8"]
    C --> R3["ground_once #3<br/>Opus 4.8"]
    R1 --> V["collect votes<br/>e.g. {SUP:2, NEI:1}"]
    R2 --> V
    R3 --> V
    V --> M["majority_label + severity tie-break<br/>(code)"]
    M --> CONF["confidence = max_votes / N<br/>3-0 → 1.0 · 2-1 → 0.67"]
    CONF --> G([GroundOutcome:<br/>label · span · rationale · votes · confidence · refused])

    style R1 fill:#fef9c3,stroke:#a16207
    style R2 fill:#fef9c3,stroke:#a16207
    style R3 fill:#fef9c3,stroke:#a16207
    style V fill:#dbeafe,stroke:#1e40af
    style M fill:#dbeafe,stroke:#1e40af
    style CONF fill:#dbeafe,stroke:#1e40af
```

---

## Caching: warm one call, then fan out

Within one check the judge sees the **same SOURCE** on every grounding call, so the SOURCE is
sent as a `cache_control: ephemeral` block. But a cache entry is only readable *after the
first response begins streaming*. So the pipeline runs a **cache gate**:

```mermaid
flowchart TD
    SRC([source]) --> EST["estimate tokens<br/>(cheap chars/4 heuristic, no API call)"]
    EST --> Q{"≥ 4096-token<br/>Opus cache floor?"}
    Q -- "yes (real RAG-sized source)" --> WARM["ground claim #1 synchronously<br/>→ warms the SOURCE cache"]
    WARM --> FAN1["fan out the rest concurrently<br/>(they read the cache ≈ 0.1× input)"]
    Q -- "no (small demo source)" --> FAN2["fan out all claims immediately<br/>(no cache benefit to wait for)"]
    FAN1 --> SAME([same report — only the scheduling differs])
    FAN2 --> SAME

    style EST fill:#dbeafe,stroke:#1e40af
    style Q fill:#dbeafe,stroke:#1e40af
    style WARM fill:#dbeafe,stroke:#1e40af
    style FAN1 fill:#dbeafe,stroke:#1e40af
    style FAN2 fill:#dbeafe,stroke:#1e40af
```

The 4096-token floor is why the small demo sources (~90 tokens) report
`cache_creation_input_tokens: 0` — that is expected, not a regression. The cache win
materializes on real RAG-sized contexts, where the N×claims grounding calls share one cached
SOURCE prefix.

---

## The provider seam

`core` talks to a model through one small `LLMProvider` Protocol (a single `parse()` method),
so the engine runs three ways without any branch in the pipeline:

```mermaid
flowchart LR
    P["LLMProvider Protocol<br/>parse(model, system, user_blocks, output_model) → ParseResult"]
    P --- A["AnthropicProvider<br/>messages.parse (Claude)"]
    P --- O["OpenAIProvider<br/>Azure gpt-5.5"]
    P --- M["MockProvider<br/>deterministic, key-free, fixture-driven"]

    SEL["get_provider()<br/>explicit arg → GROUNDCHECK_LLM env → auto-detect by key"]
    SEL --> P

    style P fill:#dbeafe,stroke:#1e40af
    style SEL fill:#dbeafe,stroke:#1e40af
    style A fill:#fef9c3,stroke:#a16207
    style O fill:#fef9c3,stroke:#a16207
    style M fill:#dcfce7,stroke:#15803d
```

- **`MockProvider`** is the default for every no-key test, CI, and the §5 worked-example
  demo. It returns canned, deterministic verdicts keyed on the (normalized) user text — and
  it can express a *split vote* as an ordered sequence of verdicts consumed by call index, so
  the 62% money demo reproduces byte-for-byte with no key.
- **`AnthropicProvider`** is the logical routing (decompose → Sonnet 4.6, ground → Opus 4.8).
- **`OpenAIProvider`** is what actually runs the live numbers in this environment, because the
  only key available here is Azure OpenAI (`gpt-5.5`). The logical Sonnet/Opus *routing* is
  unchanged; one Azure deployment serves both steps, and its cost figures are flagged as
  estimates (its list price is not in the pinned API facts).

Cost is honest by construction: every `parse()` returns a `Usage` with the three input
buckets (fresh / cache-write / cache-read) and output tokens, and `compute_cost` prices each
bucket separately. The per-check `cost_usd` includes the Sonnet decompose cost **plus** every
Opus grounding run — the decompose cost is never silently dropped.

---

## Request lifecycle (the web path)

```mermaid
sequenceDiagram
    participant B as Browser (app/)
    participant API as api/ (FastAPI adapter)
    participant E as core/ check()
    participant P as provider seam

    B->>API: POST /check {source, answer, n}
    Note over API: pydantic validates n ∈ [1,5]<br/>(422 before any engine call)
    API->>E: groundcheck.check(source, answer, n)
    E->>P: decompose (Sonnet)
    P-->>E: atomic claims
    par grounding fan-out (×N per claim)
        E->>P: ground (Opus, cached SOURCE)
        P-->>E: verdict + usage
    end
    Note over E: majority + severity · score · highlight
    E-->>API: FaithfulnessReport + captured warnings
    Note over API: map missing-key → 503, engine error → 502<br/>(never a stack trace to the client)
    API-->>B: CheckResponse (verdicts · score · highlighted_html)
```

---

## The two-tier meta-eval — the honesty headline

The hard part of any "LLM grader" is proving the grader itself is reliable. groundcheck ships
that proof as a **two-tier meta-eval** against human-labeled gold, with the two tiers split so
decomposition variance never silently corrupts a claim-level number. Full methodology,
datasets, and the τ-sweep rationale live in [`eval.md`](eval.md); the shape:

```mermaid
flowchart TD
    GOLD([gold datasets<br/>held-in + frozen ~20% slices])

    GOLD --> T1["TIER 1 — grounding-judge accuracy<br/>fixed {source, claim, gold_label} triples<br/>→ ground() 1:1 with gold (no decomposition)"]
    T1 --> M1["per-class P/R/F1 · macro-F1 · accuracy · Cohen's κ<br/>(groundcheck.metrics, pure stdlib)"]

    GOLD --> T2["TIER 2 — end-to-end answer detection<br/>full check() per answer<br/>predicted_unfaithful over τ sweep {1.0, 0.9, 0.8}"]
    T2 --> M2["binary P/R/F1 (positive = unfaithful)<br/>decomposition variance lives HERE, by design"]

    M1 --> REP["R repeats → mean ± spread<br/>(grounding is non-deterministic; never one number)"]
    M2 --> REP
    REP --> LOG[("runs/&lt;ts&gt;.jsonl<br/>header reproduces the run + per-item preds + summary")]

    style T1 fill:#dbeafe,stroke:#1e40af
    style T2 fill:#dbeafe,stroke:#1e40af
    style M1 fill:#dbeafe,stroke:#1e40af
    style M2 fill:#dbeafe,stroke:#1e40af
    style REP fill:#dbeafe,stroke:#1e40af
```

- **Tier 1** isolates the judge on fixed triples, so the detector label aligns 1:1 with gold —
  the rigorous headline. Decomposition is deliberately absent here.
- **Tier 2** runs the full pipeline and asks the answer-level question (*did we flag this as
  unfaithful?*) across a threshold sweep. Decomposition variance is folded in **here**, where
  it belongs.
- Grounding is non-deterministic, so every figure is **mean ± spread over R repeats**, never
  a single number pretending to be exact. The **frozen** slice (~20%, never tuned against) is
  reported as **accuracy + κ only** — at n≈9 the per-class F1 is too noisy to headline.

---

## Test tiers — deterministic vs live-key

The test suite mirrors the same honesty split as the eval: the bulk is **deterministic and
key-free**, and only a small set of smokes need a real key (and skip cleanly without one). It
is structurally hard for a missing key to turn a real failure into a green build.

```mermaid
flowchart TD
    RUN(["scripts/test.sh — GROUNDCHECK_LLM=mock"]) --> CORE["pytest core/tests<br/>engine + metrics + pipeline hardening"]
    RUN --> APIT["pytest api/tests<br/>incl. contract + 'no engine logic in api' grep test"]
    RUN --> EVALT["pytest eval/tests<br/>ScriptedProvider → known confusion matrix"]
    RUN --> E2E["e2e/ — Playwright money demo (asserts 62%)<br/>+ axe-core a11y audit"]

    KEY(["live-key smokes — marked @api"]) --> SKIP["skip cleanly with NO key<br/>(real provider end-to-end when a key is present)"]

    style CORE fill:#dbeafe,stroke:#1e40af
    style APIT fill:#dbeafe,stroke:#1e40af
    style EVALT fill:#dbeafe,stroke:#1e40af
    style E2E fill:#dbeafe,stroke:#1e40af
    style SKIP fill:#fef9c3,stroke:#a16207
```

The money demo (62%, three amber sentences) is asserted in `e2e/test_money_demo.py` against
the deterministic mock, so the headline figure is byte-stable and reproducible with no key.

---

## Invariants worth keeping

These are the properties the design protects; several are enforced by tests, not just
convention.

| Invariant | Why it matters | Where it's held |
|---|---|---|
| `import groundcheck` is key-free and SDK-free | the package loads fast and safe; no accidental network import | lazy SDK imports inside providers; asserted in `core/tests` |
| No service-to-service HTTP between Python components | eval stays parallel-safe; cost stays honest | everything imports `core` directly |
| No grounding/scoring/highlighting logic in `api/` | the adapter can't drift into a second engine | grep/contract test in `api/tests` |
| Split vote biases to **flag**, never certify | correct bias for a firewall | severity tie-break in `ground.py` |
| 0 claims → `None` (N/A), never 100% | an empty answer can't look perfectly faithful | `FaithfulnessReport.from_claims` |
| Refusals surfaced (`n_refused`), not hidden in the score | a safety refusal must not masquerade as a grounding result | `ground.py` → report counts |
| Per-check cost includes decompose **and** every grounding run | no silent under-counting | `pipeline.check` sums both |
| All metric math lives in `groundcheck.metrics` | one audited implementation of κ etc., reused by eval | `eval/` delegates, does no arithmetic |

---

## Repo layout

```text
core/    the engine + CLI + the metrics module (pure stdlib)    ·  from groundcheck import check
api/     FastAPI adapter over check(); serves the static app    ·  groundcheck_api.main:app
app/     the single-file dc-html demo UI (no build step)
eval/    the two-tier meta-eval harness + gold datasets         ·  python -m eval.run --tier all
scripts/ dev.* (serve the stack, no key)  ·  test.* (full no-key suite)
docs/    architecture.md (this file) · eval.md · the money-demo screenshot
```
