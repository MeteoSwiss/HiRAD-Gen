"""Mechanistic (Weight Diagnostics) evaluator."""
from .runner import run
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "mechanistic",
    "default_enabled": False,
    "scoreboard": False,
    "requires": ["checkpoint"],
}
