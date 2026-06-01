from abc import ABC, abstractmethod
import torch
import numpy as np
import mlflow
import os
import json

from hirad.distributed import DistributedManager
from hirad.utils.console import PythonLogger
from hirad.datasets import DownscalingDataset
from hirad.models import UNet, EDMPrecondSuperResolution
from hirad.utils.dataset_utils import regrid_icon_to_rotlatlon
from hirad.utils.checkpoint import load_checkpoint


class TrainingManagerBase(ABC):
    def __init__(self, dist: DistributedManager, logger: PythonLogger):
        self.dist = dist
        self.logger = logger

    @abstractmethod
    def load_and_preprocess_batch(self):
        pass

    @abstractmethod
    def get_static_data(self):
        pass

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def run_validation(self):
        pass


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
                is_real2cosmo_target: bool,
                is_real2km_target: bool,
                songunet_checkpoint_level: int,
                use_patching: bool,
                hr_mean_conditioning: bool,
                logging_method: str,
                ):
        super().__init__(dist, logger)
        self.dataset = dataset
        self.input_dtype = input_dtype
        self.img_shape = img_shape
        self.is_real_target = is_real_target
        self.is_real2cosmo_target = is_real2cosmo_target
        self.is_real2km_target = is_real2km_target
        self.n_month_hour_channels = n_month_hour_channels
        self.fp16 = fp16
        self.songunet_checkpoint_level = songunet_checkpoint_level
        self.profile_mode = profile_mode
        self.enable_amp = enable_amp
        self.amp_dtype = amp_dtype
        self.use_apex_gn = use_apex_gn
        self.use_patching = use_patching
        self.hr_mean_conditioning = hr_mean_conditioning
        self.logging_method = logging_method


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
                coarsen_by_2x=self.is_real2km_target,
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

    def load_regression_model(self, regression_checkpoint_path: str):
        """Load the regression model for the residual loss if applicable."""

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
            "profile_mode": self.profile_mode,
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
                    "augment_pipe": None,
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
                    loss_valid = (loss_valid.sum() / batch_size_per_gpu / patch_num_per_iter).cpu().item()
                    valid_loss_accum += loss_valid / validation_steps / len(patch_nums_iter)

        valid_loss_sum = torch.tensor([valid_loss_accum], device=self.dist.device)
        if self.dist.world_size > 1:
            torch.distributed.barrier()
            torch.distributed.all_reduce(valid_loss_sum, op=torch.distributed.ReduceOp.SUM)
        average_valid_loss = (valid_loss_sum / self.dist.world_size).item()
        if self.dist.rank == 0 and self.logging_method == "mlflow":
            mlflow.log_metric("validation_loss", average_valid_loss, cur_nimg)

        return average_valid_loss