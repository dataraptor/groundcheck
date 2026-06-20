# eval: the two-tier meta-eval harness

Measures how well the GroundCheck engine itself performs, against human-reviewed gold
datasets. This is offline, development-time work, separate from the shipped product: end
users never touch this code. It imports `groundcheck` **directly** (no HTTP) and delegates
**all** metric math to `groundcheck.metrics` (this layer does no P/R/F1/κ arithmetic of its
own).

The entry point is `python -m eval.run`, kept in this layer rather than in `groundcheck.cli`
so the gold datasets stay out of `core`.

**Depends on:** `core` (imported directly). `pyyaml` is an **eval-layer** dependency only,
deliberately not in `core`, so the engine stays dataset-free.

## Install

From the repo root:

```bash
python -m pip install -e ./core[dev]      # the engine (editable)
python -m pip install -r eval/requirements.txt   # pyyaml + pytest (eval-layer deps)
```

## Run

```bash
# fast, no-key smoke (mock provider; pipeline completes, numbers are not meaningful):
GROUNDCHECK_LLM=mock python -m eval.run --tier all --quick

# real run (needs a key: export Azure OpenAI creds, or ANTHROPIC_API_KEY):
python -m eval.run --tier all              # n=3, R=3  (the reported numbers)
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--tier {1,2,all}` | `all` | Tier 1 = grounding-judge accuracy on fixed claim triples; Tier 2 = end-to-end answer detection. |
| `-n N` | `3` | grounding runs per claim (the N-run majority). |
| `--repeats R` | `3` | whole-tier repeats; results are reported **mean ± spread** (grounding is non-deterministic). |
| `--quick` | off | fast iteration: forces `n=1, R=1`. |
| `--slice {held-in,frozen,all}` | `all` | report the held-in set, the frozen ~20% slice, or both separately. |
| `--out PATH` | `runs/<ts>.jsonl` | where to persist the run log. |

`--slice all` reports **held-in** (full per-class P/R/F1 + macro-F1 + accuracy + κ) and the
**frozen** slice (accuracy + κ only, with an `n≈9, wide interval` caption; the frozen set
is too small for stable per-class F1).

### Real-run provider

`get_provider()` honors `GROUNDCHECK_LLM` (`mock`/`anthropic`/`openai`) and otherwise
auto-detects by which key is present. In this repo the only real key is **Azure OpenAI
(gpt-5.5)**, in the repo-root `.env` (gitignored). The harness does **not** auto-load
`.env`; for a live run, export the creds first, e.g.:

```bash
set -a; source .env; set +a            # then `python -m eval.run --tier all`
```

A missing key surfaces the engine's clean message once and exits non-zero (no traceback).

## Datasets

- `datasets/tier1_claims.yaml`: ~45 `{source, claim, gold_label}` triples, balanced
  15/15/15 across 9 topics; ~20% frozen (stratified).
- `datasets/tier2_answers.yaml`: ~18 `{source, answer, gold_is_faithful}` cases across five
  buckets; ~20% frozen. `t2-001` / `t2-002` are byte-identical to
  `core/examples/example_hallucinated.json` / `example_grounded.json`.

Gold labels were authored 2026-06-20 and **human-reviewed** by a single annotator, so κ
measures detector↔author agreement, not inter-annotator agreement (also stated in the
project README).

## Output

Every run persists to `runs/<ts>.jsonl` (gitignored): a `run_header` line (records
`prompt_version`, both model IDs, `n`, `R`, `mock_mode`, enough to reproduce the run), one
`prediction` line per item, and a final `summary` line with the aggregated metrics plus
`refusal_affected` and `skipped` counts.

## Cost preview

A full `--tier all` run (n=3, R=3) is roughly:

- **Tier 1:** 45 triples × n3 × R3 ≈ **405** grounding (Opus/gpt-5.5) calls.
- **Tier 2:** 18 answers × R3 = 54 checks; each is 1 decompose + (≈4-6 claims × n3)
  groundings, so ≈ 54 decompose + **~800** grounding calls.

That is order **~1.2k model calls** per full run. Use `--quick` (n1/R1, ~1/9th) while
iterating, and the full run only for the reported leaderboard numbers.

## Tests

```bash
python -m pytest eval/tests -q -m "not api"   # no key; the @api smoke skips
```

The harness logic is verified with a deterministic `ScriptedProvider`
(`tests/conftest.py`) so the confusion matrix, and therefore every metric, is known in
advance and asserted exactly. The `@api` smoke runs the real provider end-to-end.
