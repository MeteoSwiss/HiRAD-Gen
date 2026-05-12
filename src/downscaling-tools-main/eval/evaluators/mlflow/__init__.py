"""MLflow training-loss evaluator."""
from .runner import run
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "mlflow",
    "default_enabled": False,
    "scoreboard": False,
    "requires": ["checkpoint"],
}
