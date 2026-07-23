from abc import ABC, abstractmethod
import torch
import numpy as np
import mlflow
import os
import json

from hirad.distributed import DistributedManager
from hirad.utils.console import PythonLogger
from hirad.datasets import DownscalingDataset
from hirad.models import UNet, EDMPrecondSuperResolution, FlowMatchingSuperResolution
from hirad.utils.dataset_utils import regrid_icon_to_rotlatlon
from hirad.utils.checkpoint import load_checkpoint


class TrainingManagerBase(ABC):
    def __init__(self, dist: DistributedManager,
                       logger: PythonLogger,
                       dataset: DownscalingDataset,
                       is_real_target: bool,
                       n_month_hour_channels: int,
                       img_shape: tuple[int, int],
                       input_dtype: torch.dtype,
                       enable_amp: bool,
                       amp_dtype: torch.dtype,
                       use_apex_gn: bool,
                       logging_method: str,
                       ):
        self.dist = dist
        self.logger = logger
        self.dataset = dataset
        self.is_real_target = is_real_target
        self.n_month_hour_channels = n_month_hour_channels
        self.img_shape = img_shape
        self.input_dtype = input_dtype
        self.logging_method = logging_method
        self.enable_amp = enable_amp
        self.amp_dtype = amp_dtype
        self.use_apex_gn = use_apex_gn

    @abstractmethod
    def create_model(self):
        pass
    
    def load_and_preprocess_batch(self, dataset_iterator):
        """Load a batch from the iterator and preprocess it (interpolate, normalize, move to device)."""
        img_clean, img_lr, *date_str = next(dataset_iterator)

        # Interpolate and normalize low-res input
        img_lr = self.dataset.interpolator(
            img_lr.to(self.dist.device, dtype=self.input_dtype)
        ).reshape(*img_lr.shape[:-1], *self.img_shape).flip(-2)
        img_lr = self.dataset.normalize_input(img_lr)

        # Process high-res target
        if self.is_real_target:
            img_clean = regrid_icon_to_rotlatlon(
                img_clean.to(self.dist.device, dtype=self.input_dtype),
                self.dataset.regrid_indices_real,
                self.dataset.regrid_weights_real,
            )
            if self.dataset.trim_edge > 0:
                img_clean = img_clean[:, :, self.dataset.trim_edge:-self.dataset.trim_edge,
                                            self.dataset.trim_edge:-self.dataset.trim_edge]
            img_clean = img_clean.flip(-2)
        else:
            img_clean = img_clean.to(self.dist.device, dtype=self.input_dtype)
            img_clean = img_clean.reshape(*img_clean.shape[:-1], *self.img_shape).flip(-2)
        img_clean = self.dataset.normalize_output(img_clean)

        # Date embedding
        date_embedding = None
        if self.n_month_hour_channels > 0:
            date_embedding = self.dataset.make_time_grids(*date_str, self.dist.device, dtype=self.input_dtype)

        # Memory format
        if self.use_apex_gn:
            img_clean = img_clean.to(self.dist.device, dtype=self.input_dtype, non_blocking=True).to(
                memory_format=torch.channels_last
            )
            img_lr = img_lr.to(self.dist.device, dtype=self.input_dtype, non_blocking=True).to(
                memory_format=torch.channels_last
            )
        else:
            img_clean = img_clean.to(self.dist.device).to(self.input_dtype).contiguous()
            img_lr = img_lr.to(self.dist.device).to(self.input_dtype).contiguous()

        return img_clean, img_lr, date_embedding

    def get_static_data(self):
        """Get static data from the dataset, preprocess it and move to device."""
        static_channels = self.dataset.get_static_data()
        if static_channels is not None:
            if isinstance(static_channels, np.ndarray):
                static_channels = torch.from_numpy(static_channels)

            static_channels = static_channels[None, ::].flip(-2)
            if self.use_apex_gn:
                static_channels = static_channels.to(
                    self.dist.device,
                    dtype=self.input_dtype,
                    non_blocking=True,
                ).to(memory_format=torch.channels_last)
            else:
                static_channels = (
                    static_channels.to(self.dist.device)
                    .to(self.input_dtype)
                    .contiguous()
                )
        return static_channels


    def run_validation(self, cur_nimg, validation_dataset_iterator, model, loss_fn, validation_steps,
                    static_channels, batch_size_per_gpu, patching,
                    patch_nums_iter, use_patch_grad_acc):
        """Run validation and return average validation loss."""
        valid_loss_accum = 0
        with torch.no_grad():
            lead_time_label_valid = None
            for _ in range(validation_steps):
                img_clean_valid, img_lr_valid, date_embedding = self.load_and_preprocess_batch(validation_dataset_iterator)

                loss_valid_kwargs = {
                    "net": model,
                    "img_clean": img_clean_valid,
                    "img_lr": img_lr_valid,
                    "static_channels": static_channels,
                    "date_embedding": date_embedding,
                    "use_apex_gn": self.use_apex_gn,
                }
                if use_patch_grad_acc is not None:
                    loss_valid_kwargs["use_patch_grad_acc"] = use_patch_grad_acc
                if use_patch_grad_acc:
                    loss_fn.y_mean = None

                for patch_num_per_iter in patch_nums_iter:
                    if patching is not None:
                        patching.set_patch_num(patch_num_per_iter)
                        loss_valid_kwargs["patching"] = patching
                    with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
                        loss_valid = loss_fn(**loss_valid_kwargs)
                    # loss_valid = (loss_valid.sum() / batch_size_per_gpu / patch_num_per_iter).cpu().item()
                    loss_valid = loss_valid.mean().cpu().item()
                    valid_loss_accum += loss_valid / validation_steps / len(patch_nums_iter)

        valid_loss_sum = torch.tensor([valid_loss_accum], device=self.dist.device)
        if self.dist.world_size > 1:
            torch.distributed.barrier()
            torch.distributed.all_reduce(valid_loss_sum, op=torch.distributed.ReduceOp.SUM)
        average_valid_loss = (valid_loss_sum / self.dist.world_size).item()
        if self.dist.rank == 0 and self.logging_method == "mlflow":
            mlflow.log_metric("validation_loss", average_valid_loss, cur_nimg)

        return average_valid_loss

    def load_regression_model(self, regression_checkpoint_path: str):
        """Load the frozen pre-trained regression model (for ResidualLoss /
        AnchoredDiffusionLoss). Lives on the base class because both the CorrDiff
        diffusion and the anchored DiT managers need it."""

        if not os.path.isdir(regression_checkpoint_path):
            raise FileNotFoundError(
                f"Expected this regression checkpoint but not found: {regression_checkpoint_path}"
            )
        #TODO make regression model loading more robust (model type is both in rergession_checkpoint_path and regression_name)
        #TODO add the option to choose epoch to load from / regression_checkpoint_path is now a folder
        regression_model_args_path = os.path.join(regression_checkpoint_path, 'model_args.json')
        if not os.path.isfile(regression_model_args_path):
            raise FileNotFoundError(f"Missing config file at '{regression_model_args_path}'.")

        with open(regression_model_args_path, 'r') as f:
            regression_model_args = json.load(f)

        regression_model_args.update({
            "use_apex_gn": self.use_apex_gn,
            "profile_mode": getattr(self, "profile_mode", False),
            "amp_mode": self.enable_amp,
        })

        regression_net = UNet(**regression_model_args)

        _ = load_checkpoint(
            path=regression_checkpoint_path,
            model=regression_net,
            device=self.dist.device
        )
        regression_net.eval().requires_grad_(False).to(self.dist.device)
        if self.use_apex_gn:
            regression_net.to(memory_format=torch.channels_last)
        self.logger.success("Loaded the pre-trained regression model")

        return regression_net


class TrainingManagerCorrDiff(TrainingManagerBase):
    def __init__(
                self, 
                dist: DistributedManager, 
                logger: PythonLogger, 
                dataset: DownscalingDataset, 
                input_dtype: torch.dtype, 
                img_shape: tuple[int, int], 
                n_month_hour_channels: int, 
                fp16: bool,
                profile_mode: bool,
                enable_amp: bool,
                amp_dtype: torch.dtype,
                use_apex_gn: bool,
                is_real_target: bool, 
                songunet_checkpoint_level: int,
                use_patching: bool,
                hr_mean_conditioning: bool,
                logging_method: str,
                ):
        super().__init__(dist,
                        logger,
                        dataset,
                        is_real_target,
                        n_month_hour_channels,
                        img_shape,
                        input_dtype,
                        enable_amp,
                        amp_dtype,
                        use_apex_gn,
                        logging_method)
        self.fp16 = fp16
        self.songunet_checkpoint_level = songunet_checkpoint_level
        self.profile_mode = profile_mode
        self.use_patching = use_patching
        self.hr_mean_conditioning = hr_mean_conditioning



    def create_model(self, cfg_model_name: str, cfg_model_args: dict, prob_channels: list = []):
        """Instantiate the model."""
        n_input_channels = len(self.dataset.input_channels())
        n_static_channels = len(self.dataset.static_channels())
        n_output_channels = len(self.dataset.output_channels())

        img_in_channels = n_input_channels + n_static_channels + self.n_month_hour_channels
        if self.hr_mean_conditioning:
            img_in_channels += n_output_channels
        if self.use_patching:
            img_in_channels += n_input_channels + n_static_channels

        img_out_channels = n_output_channels

        self.logger.info(f"Creating model {cfg_model_name} with {img_in_channels} input channels and {img_out_channels} output channels.")

        model_args = {  # default parameters for all networks
            "img_out_channels": img_out_channels,
            "img_resolution": list(self.img_shape),
            "use_fp16": self.fp16,
            "checkpoint_level": self.songunet_checkpoint_level,
        }
        if cfg_model_name == "lt_aware_ce_regression":
            model_args["prob_channels"] = prob_channels
        
        if cfg_model_args:  # override defaults from config file
            model_args.update(cfg_model_args)

        model_args["use_apex_gn"] = self.use_apex_gn
        model_args["profile_mode"] = self.profile_mode

        if self.enable_amp:
            model_args["amp_mode"] = self.enable_amp


        if cfg_model_name == "regression":
            model = UNet(
                img_in_channels=img_in_channels + model_args["N_grid_channels"],
                **model_args,
            )
            model_args["img_in_channels"] = img_in_channels + model_args["N_grid_channels"]
        elif cfg_model_name == "lt_aware_ce_regression":
            model = UNet(
                img_in_channels=img_in_channels
                + model_args["N_grid_channels"]
                + model_args["lead_time_channels"],
                **model_args,
            )
            model_args["img_in_channels"] = img_in_channels + model_args["N_grid_channels"] + model_args["lead_time_channels"]
        elif cfg_model_name == "lt_aware_patched_diffusion":
            model = EDMPrecondSuperResolution(
                img_in_channels=img_in_channels
                + model_args["N_grid_channels"]
                + model_args["lead_time_channels"],
                **model_args,
            )
            model_args["img_in_channels"] = img_in_channels + model_args["N_grid_channels"] + model_args["lead_time_channels"]
        else:  # diffusion or patched diffusion
            model = EDMPrecondSuperResolution(
                img_in_channels=img_in_channels + model_args["N_grid_channels"],
                **model_args,
            )
            model_args["img_in_channels"] = img_in_channels + model_args["N_grid_channels"]

        return model, model_args

class TrainingManagerDiT(TrainingManagerBase):
    def __init__(
                self,
                dist: DistributedManager,
                logger: PythonLogger,
                dataset: DownscalingDataset,
                input_dtype: torch.dtype,
                img_shape: tuple[int, int],
                n_month_hour_channels: int,
                fp16: bool,
                enable_amp: bool,
                amp_dtype: torch.dtype,
                is_real_target: bool,
                logging_method: str,
                use_apex_gn: bool,
                n_prev_hr_frames: int = 0,
                prev_hr_dropout: float = 0.0,
                ):
        super().__init__(dist,
                        logger,
                        dataset,
                        is_real_target,
                        n_month_hour_channels,
                        img_shape,
                        input_dtype,
                        enable_amp,
                        amp_dtype,
                        use_apex_gn,
                        logging_method)
        self.fp16 = fp16
        self.n_prev_hr_frames = n_prev_hr_frames
        self.prev_hr_dropout = prev_hr_dropout

    def create_model(self, cfg_model_name: str, cfg_model_args: dict):
        """Instantiate the model."""
        n_input_channels = len(self.dataset.input_channels())
        n_static_channels = len(self.dataset.static_channels())
        n_output_channels = len(self.dataset.output_channels())

        img_in_channels = (n_input_channels + n_static_channels
                           + self.n_prev_hr_frames * n_output_channels)
        img_out_channels = n_output_channels

        self.logger.info(
            f"Creating model {cfg_model_name} with {img_in_channels} input channels "
            f"({n_input_channels} ERA5, {n_static_channels} static, "
            f"{self.n_prev_hr_frames * n_output_channels} prev_hr) "
            f"and {img_out_channels} output channels."
        )

        model_args = {  # default parameters for all networks
            "model_type": "DiT",
            "img_in_channels": img_in_channels,
            "img_out_channels": img_out_channels,
            "img_resolution": list(self.img_shape),
            "use_fp16": self.fp16,
            "amp_mode": self.enable_amp,
            "condition_dim": self.n_month_hour_channels,
        }

        if cfg_model_args:  # override defaults from config file
            model_args.update(cfg_model_args)

        model = EDMPrecondSuperResolution(**model_args)

        return model, model_args

    def load_and_preprocess_batch(self, dataset_iterator):
        """Load a batch and preprocess it; when n_prev_hr_frames>0, also process prev HR."""
        batch = next(dataset_iterator)

        if self.n_prev_hr_frames > 0:
            img_clean, img_lr, *date_str, prev_hr_raw, prev_hr_valid = batch
        else:
            img_clean, img_lr, *date_str = batch

        # Interpolate and normalize low-res input
        img_lr = self.dataset.interpolator(
            img_lr.to(self.dist.device, dtype=self.input_dtype)
        ).reshape(*img_lr.shape[:-1], *self.img_shape).flip(-2)
        img_lr = self.dataset.normalize_input(img_lr)

        # Process high-res target
        if self.is_real_target:
            img_clean = regrid_icon_to_rotlatlon(
                img_clean.to(self.dist.device, dtype=self.input_dtype),
                self.dataset.regrid_indices_real,
                self.dataset.regrid_weights_real,
            )
            if self.dataset.trim_edge > 0:
                img_clean = img_clean[:, :,
                                       self.dataset.trim_edge:-self.dataset.trim_edge,
                                       self.dataset.trim_edge:-self.dataset.trim_edge]
            img_clean = img_clean.flip(-2)
        else:
            img_clean = img_clean.to(self.dist.device, dtype=self.input_dtype)
            img_clean = img_clean.reshape(*img_clean.shape[:-1], *self.img_shape).flip(-2)
        img_clean = self.dataset.normalize_output(img_clean)

        # Date embedding
        date_embedding = None
        if self.n_month_hour_channels > 0:
            date_embedding = self.dataset.make_time_grids(*date_str, self.dist.device, dtype=self.input_dtype)

        # Previous HR conditioning
        if self.n_prev_hr_frames > 0:
            prev_hr_raw = prev_hr_raw.to(self.dist.device, dtype=self.input_dtype)
            prev_hr_valid = prev_hr_valid.to(self.dist.device)  # (B,) bool

            # Same preprocessing as img_clean
            if self.is_real_target:
                prev_hr = regrid_icon_to_rotlatlon(
                    prev_hr_raw,
                    self.dataset.regrid_indices_real,
                    self.dataset.regrid_weights_real,
                )
                if self.dataset.trim_edge > 0:
                    prev_hr = prev_hr[:, :,
                                       self.dataset.trim_edge:-self.dataset.trim_edge,
                                       self.dataset.trim_edge:-self.dataset.trim_edge]
                prev_hr = prev_hr.flip(-2)
            else:
                prev_hr = prev_hr_raw.reshape(*prev_hr_raw.shape[:-1], *self.img_shape).flip(-2)
            prev_hr = self.dataset.normalize_output(prev_hr)

            # Force-zero boundary/gap samples (in normalized space)
            invalid_mask = (~prev_hr_valid).view(-1, 1, 1, 1)
            prev_hr = prev_hr.masked_fill(invalid_mask, 0.0)

            # Stochastic CFG dropout: only during training (grad enabled), not validation
            if self.prev_hr_dropout > 0.0 and torch.is_grad_enabled():
                keep = torch.bernoulli(
                    torch.full(
                        (prev_hr.shape[0], 1, 1, 1),
                        1.0 - self.prev_hr_dropout,
                        device=self.dist.device,
                    )
                ).to(dtype=torch.bool)
                prev_hr = prev_hr * keep

            # Append to img_lr; DiffusionLoss will later append static channels
            img_lr = torch.cat([img_lr, prev_hr], dim=1)

        # Memory format
        if self.use_apex_gn:
            img_clean = img_clean.to(self.dist.device, dtype=self.input_dtype, non_blocking=True).to(
                memory_format=torch.channels_last
            )
            img_lr = img_lr.to(self.dist.device, dtype=self.input_dtype, non_blocking=True).to(
                memory_format=torch.channels_last
            )
        else:
            img_clean = img_clean.to(self.dist.device).to(self.input_dtype).contiguous()
            img_lr = img_lr.to(self.dist.device).to(self.input_dtype).contiguous()

        return img_clean, img_lr, date_embedding


class TrainingManagerAnchoredDiT(TrainingManagerDiT):
    """DiT conditioned on a frozen regression mean (anchored / residual formulation).

    Identical to TrainingManagerDiT except:
      * the conditioning gains the regression mean's channels
        (img_in_channels += img_out_channels when hr_mean_conditioning), and
      * the anchor metadata (residual_stds, hr_mean_conditioning) is persisted in the
        returned model_args (-> model_args.json) but POPPED before constructing
        EDMPrecondSuperResolution -- it is consumed by AnchoredDiffusionLoss and the
        anchored generator, not by the network. Inference must pop these keys too.
    """

    ANCHOR_META_KEYS = ("residual_stds", "hr_mean_conditioning")

    def create_model(self, cfg_model_name: str, cfg_model_args: dict):
        """Instantiate the anchored DiT."""
        n_input_channels = len(self.dataset.input_channels())
        n_static_channels = len(self.dataset.static_channels())
        n_output_channels = len(self.dataset.output_channels())

        cfg_model_args = dict(cfg_model_args or {})
        anchor_meta = {
            k: cfg_model_args.pop(k) for k in self.ANCHOR_META_KEYS if k in cfg_model_args
        }
        if "residual_stds" not in anchor_meta:
            raise ValueError(
                "anchored DiT requires model_args.residual_stds (per-channel std of "
                "the regression residual, measured on a training-period sample)."
            )
        if len(anchor_meta["residual_stds"]) != n_output_channels:
            raise ValueError(
                f"model_args.residual_stds has {len(anchor_meta['residual_stds'])} entries "
                f"but the dataset has {n_output_channels} output channels."
            )
        hr_mean_conditioning = anchor_meta.setdefault("hr_mean_conditioning", True)

        img_in_channels = (n_input_channels + n_static_channels
                           + self.n_prev_hr_frames * n_output_channels
                           + (n_output_channels if hr_mean_conditioning else 0))
        img_out_channels = n_output_channels

        self.logger.info(
            f"Creating model {cfg_model_name} with {img_in_channels} input channels "
            f"({n_input_channels} ERA5, {n_static_channels} static, "
            f"{n_output_channels if hr_mean_conditioning else 0} regression-mean, "
            f"{self.n_prev_hr_frames * n_output_channels} prev_hr) "
            f"and {img_out_channels} output channels."
        )

        model_args = {  # default parameters for all networks
            "model_type": "DiT",
            "img_in_channels": img_in_channels,
            "img_out_channels": img_out_channels,
            "img_resolution": list(self.img_shape),
            "use_fp16": self.fp16,
            "amp_mode": self.enable_amp,
            "condition_dim": self.n_month_hour_channels,
        }
        model_args.update(cfg_model_args)

        model = EDMPrecondSuperResolution(**model_args)

        # Persist anchor metadata with the checkpoint so inference can reconstruct
        # output = regression_mean + residual_stds * D_x without manual bookkeeping.
        model_args.update(anchor_meta)

        return model, model_args


class TrainingManagerFlowMatchingDiT(TrainingManagerDiT):
    """Plain single-stage DiT trained with rectified flow matching.

    Identical to :class:`TrainingManagerDiT` except the network is wrapped in
    :class:`~hirad.models.FlowMatchingSuperResolution` (velocity prediction, no EDM
    preconditioning) instead of ``EDMPrecondSuperResolution``. Channel bookkeeping,
    data loading and preprocessing are inherited unchanged, so the flow-matching DiT
    consumes exactly the same conditioning as the EDM DiT.
    """

    def create_model(self, cfg_model_name: str, cfg_model_args: dict):
        """Instantiate the flow-matching DiT."""
        n_input_channels = len(self.dataset.input_channels())
        n_static_channels = len(self.dataset.static_channels())
        n_output_channels = len(self.dataset.output_channels())

        img_in_channels = (n_input_channels + n_static_channels
                           + self.n_prev_hr_frames * n_output_channels)
        img_out_channels = n_output_channels

        self.logger.info(
            f"Creating model {cfg_model_name} (flow matching) with {img_in_channels} "
            f"input channels ({n_input_channels} ERA5, {n_static_channels} static, "
            f"{self.n_prev_hr_frames * n_output_channels} prev_hr) "
            f"and {img_out_channels} output channels."
        )

        model_args = {  # default parameters for all networks
            "model_type": "DiT",
            "img_in_channels": img_in_channels,
            "img_out_channels": img_out_channels,
            "img_resolution": list(self.img_shape),
            "use_fp16": self.fp16,
            "amp_mode": self.enable_amp,
            "condition_dim": self.n_month_hour_channels,
        }

        if cfg_model_args:  # override defaults from config file
            model_args.update(cfg_model_args)

        model = FlowMatchingSuperResolution(**model_args)

        return model, model_args
