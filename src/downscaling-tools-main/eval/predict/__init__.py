"""eval.predict — Modular prediction generation from input bundles.

Replaces generate_predictions_25_files.py with clean module boundaries.
See README.md for usage.
"""

from .bundle_manager import BUNDLE_RE, date_str_to_datetime64, discover_bundles, parse_bundle_key, resolve_date_step_pairs
from .dataset_builder import build_prediction_dataset, validate_predictions_dataset
from .distributed_io import Rank0FileWriter, _destroy_process_group, _distributed_barrier
from .inference_engine import predict_ensemble_members, predict_single_bundle
from .main import create_parser, main
from .model_loader import load_inference_model, setup_distributed
from .output_writer import prediction_output_path, write_predictions_file
from .types import BundleKey, EnsemblePrediction, PredictionConfig, PredictionMetadata, PredictionResult

__all__ = [
    "BUNDLE_RE",
    "BundleKey",
    "EnsemblePrediction",
    "PredictionConfig",
    "PredictionMetadata",
    "PredictionResult",
    "Rank0FileWriter",
    "_destroy_process_group",
    "_distributed_barrier",
    "build_prediction_dataset",
    "create_parser",
    "date_str_to_datetime64",
    "discover_bundles",
    "load_inference_model",
    "main",
    "parse_bundle_key",
    "predict_ensemble_members",
    "predict_single_bundle",
    "prediction_output_path",
    "resolve_date_step_pairs",
    "setup_distributed",
    "validate_predictions_dataset",
    "write_predictions_file",
]
