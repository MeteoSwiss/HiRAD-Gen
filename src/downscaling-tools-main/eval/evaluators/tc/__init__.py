"""TC (Tropical Cyclone Extremes) evaluator."""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "tc",
    "default_enabled": True,
    "scoreboard": True,
    "requires": ["predictions"],
}
