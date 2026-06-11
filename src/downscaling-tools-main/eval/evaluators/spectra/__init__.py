"""Spectra (Power Spectra) evaluator."""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "spectra",
    "default_enabled": True,
    "scoreboard": True,
    "requires": ["predictions"],
}
