"""Intermediate (Diffusion Step Visualization) evaluator."""
from .runner import run
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "intermediate",
    "default_enabled": False,
    "scoreboard": False,
    "requires": ["predictions"],
}
