"""Sigma (Noise Schedule Sweeps) evaluator."""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "sigma",
    "default_enabled": True,
    "scoreboard": True,
    "requires": ["predictions", "checkpoint"],
}
