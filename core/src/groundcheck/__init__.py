"""groundcheck — the Faithfulness Firewall engine.

Importing this package must stay key-free and fast: it pulls in only the data
contracts and the canonical prompts, never ``anthropic`` or anything network-y.
The LLM client, decompose/ground/pipeline steps, and CLI arrive in later splits and
lazy-import their network dependencies inside functions.
"""

from .decompose import DecomposeOutcome, decompose
from .ground import (
    GroundOutcome,
    confidence,
    ground,
    ground_once,
    majority_label,
)
from .highlight import highlight_answer
from .metrics import (
    accuracy,
    binary_prf,
    cohen_kappa,
    confusion_matrix,
    macro_f1,
    per_class_prf,
    tier1_report,
)
from .models import (
    ClaimResult,
    DecomposedClaim,
    Decomposition,
    FaithfulnessReport,
    GroundingVerdict,
    Label,
)
from .pipeline import check
from .prompts import PROMPT_VERSION

__all__ = [
    "Label",
    "DecomposedClaim",
    "Decomposition",
    "GroundingVerdict",
    "ClaimResult",
    "FaithfulnessReport",
    "PROMPT_VERSION",
    "decompose",
    "DecomposeOutcome",
    "ground",
    "ground_once",
    "majority_label",
    "confidence",
    "GroundOutcome",
    "check",
    "highlight_answer",
    "confusion_matrix",
    "per_class_prf",
    "macro_f1",
    "accuracy",
    "cohen_kappa",
    "tier1_report",
    "binary_prf",
]
