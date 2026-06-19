"""GroundCheck meta-eval layer (spec §4/§11).

This package is the offline evaluation harness. It imports ``groundcheck``
**directly** (no HTTP) and measures the detector against human-reviewed gold
datasets — Tier-1 (grounding-judge accuracy on fixed claim triples) and Tier-2
(end-to-end answer-level detection). It is its own layer, separate from ``core``,
so ``core`` stays a pure, dataset-free, independently-installable engine
(``pyyaml`` is an eval-layer dependency only).

Run it with ``python -m eval.run`` (see :mod:`eval.run`).
"""
