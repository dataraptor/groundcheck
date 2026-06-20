# docs

Design documentation for groundcheck. Start with the [root README](../README.md) for the
pitch, the money demo, and the leaderboard; come here for how it's built and how it's
measured.

| Doc | What's in it |
|---|---|
| [`architecture.md`](architecture.md) | The full system: component wiring (no service-to-service HTTP), the `check()` pipeline with its honesty rails, N-run majority, the caching gate, the provider seam, the request lifecycle, and the design invariants. |
| [`eval.md`](eval.md) | The two-tier meta-eval methodology — why two tiers, the metrics, the Tier-2 τ-sweep, distributional reporting, held-in vs frozen slices, the datasets, and reproducibility. |
| `money-demo.png` | The screenshot used in the root README (62%, three amber sentences). |

All diagrams use one legend: 🟦 deterministic code · 🟨 LLM (distributional) · 🟥 honesty rail
(degradation / N-A short-circuit).
