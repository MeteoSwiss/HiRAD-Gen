"""Capture golden records for one (setup, stage) by running the real entry point with hooks.

    python -m tests.golden.capture --setup era_real_cv1 --stage generate --out /path/to/run

Stages: ``generate`` (inference end to end, incl. the torch/GRIB files it writes),
``train_regression`` / ``train_diffusion`` (a few optimizer steps warm-started from the
production checkpoint). Run each (setup, stage) in its own process: the entry points keep
process-global state (Hydra, DistributedManager, torch._dynamo).

``--group-norm`` selects the GroupNorm implementation. ``apex`` (default) is what the
production checkpoints were trained with and what inference uses; its kernels are not bitwise
reproducible, so these captures are compared with tolerances. ``torch`` runs the same weights
with the PyTorch implementation under strict deterministic algorithms; those captures are
bitwise reproducible and compared exactly.
"""

import argparse
import importlib.util
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
os.environ.setdefault("HYDRA_FULL_ERROR", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from .recorder import Recorder  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
# The hirad package on PYTHONPATH (normally <repo>/src/hirad; another source tree, e.g. an export
# of an older commit, can be captured with this harness by putting it first on PYTHONPATH). Its
# Hydra configs are used, so code and configs always come from the same tree.
HIRAD_DIR = Path(importlib.util.find_spec("hirad").submodule_search_locations[0]).resolve()
CONFIG_DIR = HIRAD_DIR / "conf"
DEFAULT_REGISTRY = Path(__file__).with_name("registry.yaml")
STAGES = ("generate", "train_regression", "train_diffusion")
GROUP_NORM_MODES = ("apex", "torch")
# Hydra override selecting the PyTorch GroupNorm (checkpoints are converted on load).
TORCH_GROUP_NORM_OVERRIDE = {"generate": "++generation.perf.use_apex_gn=False",
                             "train": "++training.perf.use_apex_gn=False"}


def load_registry(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _quoted_list(values) -> str:
    return "[" + ",".join(f"'{v}'" for v in values) + "]"


def generate_argv(setup: dict, common: dict, run_dir: Path) -> list[str]:
    stage = setup["generate"]
    checkpoints = setup["checkpoints"]
    return [
        "generate.py",
        f"--config-path={CONFIG_DIR}",
        f"--config-name={stage['config']}",
        f"hydra.run.dir={run_dir}",
        f"generation.io.res_ckpt_path={checkpoints['diffusion']}",
        f"generation.io.reg_ckpt_path={checkpoints['regression']}",
        "generation.io.output_path=.",
        f"++generation.io.output_format={stage.get('output_format', 'both')}",
        f"++generation.io.grib_template_path={common['grib_template_path']}",
        f"generation.num_ensembles={stage.get('num_ensembles', common['num_ensembles'])}",
        f"generation.seed_batch_size={stage.get('seed_batch_size', common['seed_batch_size'])}",
        "++generation.randomize=True",
        f"++generation.random_seed={common['random_seed']}",
        "generation.times_range=null",
        f"generation.times={_quoted_list(stage['times'])}",
        "generation.perf.use_torch_compile=False",
        "++generation.perf.enable_timing=False",
        "++generation.perf.num_writer_workers=2",
        *setup.get("overrides", []),
        *stage.get("overrides", []),
    ]


def train_argv(setup: dict, common: dict, model: str, run_dir: Path) -> list[str]:
    stage = setup["train"][model]
    checkpoints = setup["checkpoints"]
    argv = [
        "train.py",
        f"--config-path={CONFIG_DIR}",
        f"--config-name={stage['config']}",
        f"hydra.run.dir={run_dir}",
        "logging.method=null",
        "training.hp.batch_size_per_gpu=1",
        "training.hp.total_batch_size=auto",
        f"training.hp.training_duration={common['train_steps']}",
        "training.perf.torch_compile=False",
        "training.perf.dataloader_workers=0",
        "training.io.print_progress_freq=1",
        "training.io.save_checkpoint_freq=1000000000",
        "++training.io.checkpoint_dir=.",
        "++dataset.validation=False",
        f"++training.io.finetune_checkpoint_path={checkpoints[model]}",
    ]
    if model == "diffusion":
        argv.append(f"++training.io.regression_checkpoint_path={checkpoints['regression']}")
    return argv + setup.get("overrides", []) + stage.get("overrides", [])


def make_deterministic(strict: bool) -> None:
    """``strict`` makes any operation without a deterministic implementation raise instead of
    warn (apex kernels are not covered by this check, hence only for the torch mode)."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def record_saved_checkpoint(rec: Recorder, run_dir: Path, production_dir: str) -> None:
    """Record what train.py wrote (model_args.json, state-dict layout) and drop the large files."""
    checkpoint_dirs = sorted(run_dir.glob("checkpoints_*"))
    if len(checkpoint_dirs) != 1:
        raise RuntimeError(f"expected one checkpoints_* dir in {run_dir}, found {checkpoint_dirs}")
    checkpoint_dir = checkpoint_dirs[0]
    rec.json("saved/checkpoint_dir_name", checkpoint_dir.name)
    with open(checkpoint_dir / "model_args.json") as f:
        rec.json("saved/model_args", json.load(f))
    with open(Path(production_dir) / "model_args.json") as f:
        rec.json("production/model_args", json.load(f))
    files = sorted(p.name for p in checkpoint_dir.glob("*.pt"))
    rec.json("saved/files", files)
    for path in checkpoint_dir.glob("*.pt"):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if path.name.startswith("checkpoint."):
            rec.json("saved/training_state_keys", sorted(state.keys()))
            rec.json("saved/epoch", state.get("epoch"))
        else:
            keys = sorted(state.keys())
            rec.json("saved/state_dict_layout", [[k, list(state[k].shape), str(state[k].dtype)] for k in keys])
            rec.array("saved/param_mean", np.array([state[k].double().mean().item() for k in keys]))
            rec.array("saved/param_abs_mean", np.array([state[k].double().abs().mean().item() for k in keys]))
        path.unlink()


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--out", required=True, type=Path, help="capture root; writes <out>/<setup>/<stage>/")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--group-norm", choices=GROUP_NORM_MODES, default="apex",
                        help="GroupNorm implementation (torch: bitwise reproducible, compared exactly)")
    parser.add_argument("--override", action="append", default=[],
                        help="extra Hydra override appended last (repeatable), e.g. for negative controls")
    args = parser.parse_args()

    registry = load_registry(args.registry)
    setup = registry["setups"][args.setup]
    common = registry["common"]
    if args.stage != "generate" and args.stage.removeprefix("train_") not in setup.get("train", {}):
        print(f"[golden] {args.setup} has no {args.stage} stage; skipping")
        return
    out_dir = args.out.resolve() / args.setup / args.stage
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty; refusing to overwrite a capture")
    run_dir = out_dir / "run"
    rec = Recorder(out_dir)
    make_deterministic(strict=args.group_norm == "torch")

    if args.stage == "generate":
        from .hooks import install_generate_hooks
        install_generate_hooks(rec)
        from hirad.inference import generate as entry
        argv = generate_argv(setup, common, run_dir)
    else:
        model = args.stage.removeprefix("train_")
        from .hooks import install_train_hooks
        install_train_hooks(rec)
        from hirad.training import train as entry
        argv = train_argv(setup, common, model, run_dir)
    if args.group_norm == "torch":
        argv.append(TORCH_GROUP_NORM_OVERRIDE["generate" if args.stage == "generate" else "train"])
    argv += args.override

    manifest = {
        "setup": args.setup,
        "stage": args.stage,
        "group_norm": args.group_norm,
        "registry_entry": setup,
        "argv": argv,
        "hirad_source": str(HIRAD_DIR),
        # git state of this repo; describes the code only when hirad_source is <repo>/src/hirad.
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty_paths": _git("status", "--porcelain", "--", "src").splitlines(),
        "hostname": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "status": "failed",
    }
    start_cwd, start_time = os.getcwd(), time.time()
    sys.argv = argv
    try:
        entry.main()
        rec.step = 0
        if args.stage != "generate":
            record_saved_checkpoint(rec, run_dir, setup["checkpoints"][args.stage.removeprefix("train_")])
        manifest["status"] = "ok"
    finally:
        os.chdir(start_cwd)
        manifest["duration_s"] = round(time.time() - start_time, 1)
        rec.save(manifest)
        print(f"[golden] {args.setup}/{args.stage}: {manifest['status']} in {manifest['duration_s']} s, "
              f"{len(rec.index)} records -> {out_dir}")


if __name__ == "__main__":
    main()
