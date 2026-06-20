# core: the engine

The faithfulness verifier itself, as a framework-free, installable Python package. This is
the bottom of the stack: it knows nothing about HTTP, the UI, or how it's deployed.

`import groundcheck` loads no SDK and needs no API key (providers import lazily), so the
engine can drop into a script, a notebook, the API layer, or the eval harness without a
server running. The public surface is one call:

```python
from groundcheck import check
report = check(source, answer, n=3)
```

It also ships a CLI (`python -m groundcheck.cli check ...`) and the `groundcheck.metrics`
module (pure stdlib: precision, recall, F1, Cohen's κ), which the eval harness reuses so
all the metric math lives in one place.

**Contains:** the package source (`src/`), worked examples (`examples/`), its
`pyproject.toml`, and unit tests.

**Depends on:** nothing else in this repo.

## Install and test

```bash
python -m pip install -e "./core[dev]"
python -m pytest core/tests -q
```
