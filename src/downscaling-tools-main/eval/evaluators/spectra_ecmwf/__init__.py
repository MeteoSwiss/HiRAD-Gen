"""ECMWF spectra evaluator — gptosp pipeline (AC-only)."""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "spectra_ecmwf",
    "default_enabled": False,
    "scoreboard": False,
    "requires": ["predictions"],
    "host_constraint": "ac",
}
