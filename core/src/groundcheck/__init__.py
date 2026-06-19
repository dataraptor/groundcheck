"""groundcheck — the Faithfulness Firewall engine.

Importing this package must stay key-free and fast: it pulls in only the data
contracts and the canonical prompts, never ``anthropic`` or anything network-y.
The LLM client, decompose/ground/pipeline steps, and CLI arrive in later splits and
lazy-import their network dependencies inside functions.
"""

from .models import (
    ClaimResult,
    DecomposedClaim,
    Decomposition,
    FaithfulnessReport,
    GroundingVerdict,
    Label,
)
from .prompts import PROMPT_VERSION

__all__ = [
    "Label",
    "DecomposedClaim",
    "Decomposition",
    "GroundingVerdict",
    "ClaimResult",
    "FaithfulnessReport",
    "PROMPT_VERSION",
]
