"""Recording hooks installed around the real entry points (``generate.py`` / ``train.py``).

Hooks wrap library functions/classes at runtime; nothing in ``src/`` is modified. Records are
named semantically so references stay comparable while internals are refactored. When a
refactor renames or reshapes one of the wrapped interfaces, update the hook here, not the
references.
"""

from pathlib import Path

import numpy as np
import torch

from hirad.datasets import get_strings_from_channels
from hirad.losses import RegressionLoss, ResidualLoss
from hirad.models import EDMPrecondSuperResolution, SongUNet, UNet
from hirad.utils.dataset_utils import regrid_icon_to_rotlatlon

from .recorder import Recorder, to_numpy

BATCH_FIELDS = ("img_clean", "img_lr", "date_embedding", "lead_time_label", "prev_hr", "has_prev")


def _require_finite(name: str, value) -> None:
    """References built from NaN data (e.g. a purged store, which anemoi reads as NaN) are useless."""
    if value is not None and not torch.isfinite(torch.as_tensor(value)).all():
        raise RuntimeError(f"golden capture: non-finite values in {name}")


def _probe(n_points: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((2, n_points)).astype(np.float32)


def fingerprint_dataset(rec: Recorder, dataset, prefix: str, sample_positions=()) -> None:
    """Behaviour of the dataset as seen through its public interface (before stats_to_torch)."""
    rec.json(f"{prefix}/type", type(dataset).__name__)
    rec.json(f"{prefix}/len", len(dataset))
    rec.json(f"{prefix}/time", list(dataset.time()))
    if hasattr(dataset, "base_time"):
        rec.json(f"{prefix}/base_time", list(dataset.base_time()))
    rec.json(f"{prefix}/input_channels", get_strings_from_channels(dataset.input_channels()))
    rec.json(f"{prefix}/output_channels", get_strings_from_channels(dataset.output_channels()))
    rec.json(f"{prefix}/static_channels", get_strings_from_channels(dataset.static_channels()))
    rec.json(f"{prefix}/image_shape", list(dataset.image_shape()))
    rec.json(f"{prefix}/provides", {
        "lead_time": bool(getattr(dataset, "provides_lead_time", False)),
        "prev_target": bool(getattr(dataset, "provides_prev_target", False)),
    })
    for attr in ("input_mean", "input_std", "output_mean", "output_std"):
        rec.array(f"{prefix}/{attr}", getattr(dataset, attr))
    rec.array(f"{prefix}/static", dataset.get_static_data())
    rec.json(f"{prefix}/previous_index", [dataset.previous_index(i) for i in range(len(dataset))])

    n_input_points = len(dataset.interpolator.longitudes_orig)
    rec.array(f"{prefix}/interpolation_probe", dataset.interpolator(_probe(n_input_points, seed=0)))
    if getattr(dataset, "real_target", False):
        n_icon = int(dataset.regrid_indices_real.max()) + 1
        icon_probe = torch.from_numpy(_probe(n_icon, seed=1))
        rec.array(f"{prefix}/regrid_probe", regrid_icon_to_rotlatlon(
            icon_probe, dataset.regrid_indices_real, dataset.regrid_weights_real))

    for j, pos in enumerate(sample_positions):
        target, inputs, date, *extras = dataset[pos]
        name = f"{prefix}/raw_sample{j}"
        record = rec.array if j == 0 else rec.summary
        rec.json(f"{name}/position", int(pos))
        rec.json(f"{name}/date", date)
        record(f"{name}/target", target[None])
        record(f"{name}/input", inputs[None])
        rec.json(f"{name}/n_extras", len(extras))
        if getattr(dataset, "provides_prev_target", False):
            prev_target, has_prev = extras[-2:]
            rec.summary(f"{name}/prev_target", prev_target)
            rec.json(f"{name}/has_prev", int(has_prev))
            extras = extras[:-2]
        if getattr(dataset, "provides_lead_time", False):
            rec.json(f"{name}/lead_time_label", int(extras[-1]))
        record(f"{name}/normalized_input", dataset.normalize_input(to_numpy(inputs)[None].copy()))
        record(f"{name}/normalized_target", dataset.normalize_output(to_numpy(target)[None].copy()))


def install_network_hooks(rec: Recorder) -> None:
    """Record the first call per step of each network: the exact first-conv input of the U-Net
    (conditioning contract incl. positional/lead-time embeddings) and the network output."""
    unet_forward = UNet.forward
    edm_forward = EDMPrecondSuperResolution.forward
    song_forward = SongUNet.forward

    def recording_unet_forward(self, x, img_lr, force_fp32=False, **kwargs):
        with rec.in_context("regression"):
            out = unet_forward(self, x, img_lr, force_fp32=force_fp32, **kwargs)
        if not rec.has("regression/net_output"):
            rec.summary("regression/net_output", out)
        return out

    def recording_edm_forward(self, x, img_lr, sigma, force_fp32=False, **kwargs):
        with rec.in_context("diffusion"):
            out = edm_forward(self, x, img_lr, sigma, force_fp32=force_fp32, **kwargs)
        if not rec.has("diffusion/net_output"):
            rec.json("diffusion/sigma", to_numpy(sigma).reshape(-1).astype(float).tolist())
            rec.summary("diffusion/net_output", out)
        return out

    def recording_song_forward(self, x, noise_labels, class_labels, augment_labels=None):
        name = f"{rec.context}/unet_input"
        if not rec.has(name):
            rec.summary(name, x)
            rec.json(f"{rec.context}/noise_labels", to_numpy(noise_labels).reshape(-1).astype(float).tolist())
        return song_forward(self, x, noise_labels, class_labels, augment_labels)

    UNet.forward = recording_unet_forward
    EDMPrecondSuperResolution.forward = recording_edm_forward
    SongUNet.forward = recording_song_forward


def install_generate_hooks(rec: Recorder) -> None:
    from hirad.inference import generate as generate_module

    install_network_hooks(rec)

    get_dataset = generate_module.get_dataset_and_sampler_inference

    def recording_get_dataset(*args, **kwargs):
        dataset, sampler = get_dataset(*args, **kwargs)
        rec.json("dataset/requested_positions", [int(i) for i in sampler])
        fingerprint_dataset(rec, dataset, "dataset", sample_positions=sampler)
        return dataset, sampler

    generate_module.get_dataset_and_sampler_inference = recording_get_dataset

    class RecordingGenerator(generate_module.Generator):
        def generate(self, image_lr, static_channels=None, date_embedding=None, lead_time_label=None,
                     prev_hr=None, **kwargs):
            rec.step += 1
            # Full arrays for the first step; hash + statistics afterwards (the written run
            # files hold the full outputs of every step).
            record = rec.array if rec.step == 1 else rec.summary
            record("inputs/image_lr", image_lr)
            if rec.step == 1:
                rec.array("inputs/static_channels", static_channels)
            rec.array("inputs/date_embedding", date_embedding)
            rec.array("inputs/lead_time_label", lead_time_label)
            if prev_hr is None:
                rec.json("inputs/prev_hr", None)
            else:
                record("inputs/prev_hr", prev_hr)
            _require_finite("inputs/image_lr", image_lr)
            image_out, image_reg, image_out_local = super().generate(
                image_lr, static_channels=static_channels, date_embedding=date_embedding,
                lead_time_label=lead_time_label, prev_hr=prev_hr, **kwargs)
            _require_finite("outputs/image_out", image_out)
            rec.json("seeds", [batch.tolist() for batch in self.rank_batches])
            rec.summary("outputs/image_out", image_out)
            if image_reg is not None:
                rec.summary("outputs/image_reg", image_reg)
            if image_out_local is not image_out:
                rec.summary("outputs/image_out_local", image_out_local)
            return image_out, image_reg, image_out_local

    generate_module.Generator = RecordingGenerator


def install_train_hooks(rec: Recorder) -> None:
    from hirad.training import train as train_module
    from hirad.training.training_manager import TrainingManagerCorrDiff
    from hirad.utils.checkpoint import _get_checkpoint_filename, match_group_norm_layout

    install_network_hooks(rec)

    configure_cuda = train_module.configure_cuda_for_consistent_precision

    def deterministic_configure_cuda():
        configure_cuda()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    train_module.configure_cuda_for_consistent_precision = deterministic_configure_cuda

    init_datasets = train_module.init_train_valid_datasets_from_config

    def recording_init_datasets(*args, **kwargs):
        result = init_datasets(*args, **kwargs)
        fingerprint_dataset(rec, result[0], "train_dataset")
        return result

    train_module.init_train_valid_datasets_from_config = recording_init_datasets

    load_batch = TrainingManagerCorrDiff.load_and_preprocess_batch

    def recording_load_batch(self, iterator):
        batch = load_batch(self, iterator)
        rec.step += 1
        for name, value in zip(BATCH_FIELDS, batch):
            if rec.step == 1 or value is None:
                rec.array(f"batch/{name}", value)
            else:
                rec.summary(f"batch/{name}", value)
        return batch

    TrainingManagerCorrDiff.load_and_preprocess_batch = recording_load_batch

    def wrap_loss(cls, label):
        call = cls.__call__

        def recording_call(self, *args, **kwargs):
            loss = call(self, *args, **kwargs)
            _require_finite(f"loss/{label}", loss.detach())
            if not rec.has(f"loss/{label}_sum"):
                patching = kwargs.get("patching")
                if patching is not None:
                    rec.json("loss/patch_indices", [list(map(int, p)) for p in patching.patch_indices])
                rec.json(f"loss/{label}_sum", float(loss.detach().double().sum()))
                rec.summary(f"loss/{label}_tensor", loss)
            return loss

        cls.__call__ = recording_call

    wrap_loss(RegressionLoss, "regression")
    wrap_loss(ResidualLoss, "residual")

    noise_params = ResidualLoss.get_noise_params

    def recording_noise_params(self, y):
        n, sigma, weight = noise_params(self, y)
        if not rec.has("loss/sigma"):
            rec.json("loss/sigma", to_numpy(sigma).reshape(-1).astype(float).tolist())
        return n, sigma, weight

    ResidualLoss.get_noise_params = recording_noise_params

    update_lr = train_module.update_learning_rate

    def recording_update_lr(*args, **kwargs):
        lr = update_lr(*args, **kwargs)
        rec.json("learning_rate", float(lr))
        return lr

    train_module.update_learning_rate = recording_update_lr

    handle_gradients = train_module.handle_and_clip_gradients

    def recording_handle_gradients(model, grad_clip_threshold=None):
        group_sq, total_sq, n_with_grad = {}, 0.0, 0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            n_with_grad += 1
            sq = float(param.grad.detach().double().pow(2).sum())
            parts = name.split(".")
            group = parts[1] if parts[0] == "model" and len(parts) > 1 else parts[0]
            group_sq[group] = group_sq.get(group, 0.0) + sq
            total_sq += sq
        rec.json("grad/total_norm", total_sq ** 0.5)
        rec.json("grad/group_norms", {k: v ** 0.5 for k, v in sorted(group_sq.items())})
        rec.json("grad/n_params_with_grad", n_with_grad)
        return handle_gradients(model, grad_clip_threshold=grad_clip_threshold)

    train_module.handle_and_clip_gradients = recording_handle_gradients

    load_partial = train_module.load_model_weights_partial

    def recording_load_partial(path, model, device="cpu"):
        module = getattr(model, "module", model)
        file_name = _get_checkpoint_filename(path, module.__class__.__name__)
        # Keys as the loader sees them (GroupNorm parameters moved to the model's layout).
        pretrained = match_group_norm_layout(torch.load(file_name, map_location="cpu"), module)
        current = module.state_dict()
        rec.json("init/checkpoint_file", Path(file_name).name)
        rec.json("init/missing_in_checkpoint", sorted(k for k in current if k not in pretrained))
        rec.json("init/unexpected_in_checkpoint", sorted(k for k in pretrained if k not in current))
        rec.json("init/shape_mismatch", sorted(
            k for k in current if k in pretrained and tuple(pretrained[k].shape) != tuple(current[k].shape)))
        return load_partial(path, model, device)

    train_module.load_model_weights_partial = recording_load_partial
