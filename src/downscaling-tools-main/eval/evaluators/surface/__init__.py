"""Surface (nMSE) evaluator."""
from .runner import run
from .scorer import score

EVALUATOR_SPEC = {
    "name": "surface",
    "default_enabled": True,
    "scoreboard": True,
    "requires": ["predictions"],
}
