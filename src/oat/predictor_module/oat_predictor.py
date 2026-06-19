from typing import Any, Dict, Literal, Type, Optional, List
from importlib import import_module
import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from omegaconf import DictConfig

from oat.model_module import model
from oat.model_module.model import (
    OATConfig,
    OAT,
    UninformedOATConfig,
    UninformedOAT,
    get_model_cls,
)
from oat.predictor_module.schedulers import estimate_total_steps
from oat.data_module.transform import Transform
from oat.data_module.support import VariableSupport
from oat.predictor_module.base_predictor import BasePredictor
from oat.predictor_module.functional_metrics import (
    sample_mae, 
    point_mae,
    mc_crps
)

logger = logging.getLogger(__name__)

class OATPredictor(BasePredictor):
    
    """
    PyTorch Lightning module for training and evaluating an OAT-based model
    
        Args:
        oat_model_cfg:
            Configuration used to construct the underlying OAT model.

        input_transform:
            Transformation applied to model inputs before they are passed to
            the network (e.g., normalization or feature preprocessing).

        output_transform:
            Transformation applied to model outputs or targets (e.g inverse-normalization).

        output_var_names:
            Names of the predicted output variables. Used for logging,
            metric tracking, or structured output handling.

        lr:
            Learning rate used to initialize the optimizer. Defaults to 1e-3.

        optimizer_class:
            Optimizer class used during training. Defaults to Adam.

        optimizer_params:
            Additional keyword arguments passed to the optimizer during
            initialization. Must match the signature of the torch optimizer
            passed as `optimizer_class`
            
        lr_scheduler_class:
            Learning rate scheduler applied to the optimizer. If None,
            no scheduler is used.

        lr_scheduler_params:
            Parameters passed to the learning rate scheduler during
            initialization. Must match the signature of the torch scheduler
            passed as `lr_scheduler_class`

        oat_loss_scheduler_class:
            Optional scheduler controlling the weighting of the OAT loss between
            "pre-oat" (before conditioning on observations) and "post-oat" (after
            conditioning) during training.

        oat_loss_scheduler_params:
            Parameters used to initialize the OAT loss weight scheduler.
            The dictionary must contain:
            start_weight:
                Initial loss weight (float between 0.0 and 1.0) used during the warmup phase and as the
                starting value for the linear schedule.
            flat_warmup_steps:
                Number of initial steps during which the weight remains constant
                at `start_weight`.
            end_weight:
                Final loss weight (float between 0.0 and 1.0) reached at `total_steps`.
            total_steps (optional):
                Total number of scheduler steps, including the warmup phase. 
                If not given in the config, it will be estimated before instantiating this class
                in train_model.py and passed to the scheduler's parameters.

    """
    def __init__(self,
                 oat_model_cfg: OATConfig | UninformedOATConfig,
                 input_transform: Transform,
                 output_transform: Transform,
                 output_var_names: List,
                 lr: float = 1e-3,
                 optimizer_class: Type[Optimizer] = torch.optim.Adam,
                 optimizer_params: Optional[dict] = None,
                 lr_scheduler_class: Optional[Type[LRScheduler]] = None,
                 lr_scheduler_params: Optional[dict] = None,
                 oat_loss_scheduler_class: Optional[Type[nn.Module]] = None,
                 oat_loss_scheduler_params:  Optional[dict]  = None,
                 ):
        super().__init__(output_var_names=output_var_names)

        self.save_hyperparameters()

        self.optimizer_class = optimizer_class
        self.optimizer_params = {"lr": lr, **(optimizer_params or {})}

        self.lr_scheduler_class = lr_scheduler_class
        self.lr_scheduler_params = lr_scheduler_params or {}

        if isinstance(oat_model_cfg, OATConfig):
            self.model = OAT(oat_model_cfg)
        elif isinstance(oat_model_cfg, UninformedOATConfig):
            self.model = UninformedOAT(oat_model_cfg)

        if not hasattr(self, "model"):
            raise ValueError(f"oat_model_cfg must be an instance of OATConfig or UninformedOATConfig. Given {type(oat_model_cfg)}")

        self.input_transform = input_transform
        self.output_transform = output_transform
        
        # NOTE perhaps make this configurable?
        self.metrics_fn = {
            # 'sample_mae': sample_mae,
            'point_mae': point_mae,
            'crps': mc_crps,
        }
        
        if oat_loss_scheduler_class is not None:
            assert oat_loss_scheduler_params is not None
            assert isinstance(oat_model_cfg, OATConfig)
            self.oat_loss_weight_scheduler = oat_loss_scheduler_class(**oat_loss_scheduler_params)
            logger.info(f"OAT loss scheduler {oat_loss_scheduler_class} initialized with parameters: {oat_loss_scheduler_params}")
        else:
            logger.info(f"No OAT loss scheduler configured. Skipping initialization.")
            self.oat_loss_weight_scheduler = None

        assert list(self.model.output_head.variable_to_distribution.keys()) == list(self.output_var_names)

    def _format_support_interval(self, support: VariableSupport, idx: int) -> str:
        lo = float(support.lower_bounds[idx].item())
        hi = float(support.upper_bounds[idx].item())
        lo_bracket = "(" if bool(support.lower_open[idx].item()) else "["
        hi_bracket = ")" if bool(support.upper_open[idx].item()) else "]"
        return f"{lo_bracket}{lo}, {hi}{hi_bracket}"

    def _format_distribution_support(self, dist_support: Any) -> str:
        if dist_support is torch.distributions.constraints.nonnegative:
            return "nonnegative"
        return str(dist_support)

    def validate_output_head_supports(self, training_dataset: Any) -> None:
        """Validate that each output distribution support covers transformed training support."""
        if (
            not hasattr(training_dataset, "dataset")
            or not hasattr(training_dataset.dataset, "obs_var_support")
            or not hasattr(training_dataset.dataset, "config")
            or not hasattr(training_dataset.dataset.config, "obs_vars")
        ):
            raise TypeError(
                "Expected a TrainDataset-like object with `.dataset.obs_var_support` "
                "and `.dataset.config.obs_vars` "
                "(from oat.data_module.data_split.TrainDataset)."
            )

        obs_vars = list(training_dataset.dataset.config.obs_vars)
        transformed_support = self.output_transform.transform_support(
            training_dataset.dataset.obs_var_support
        )

        for i, var_name in enumerate(obs_vars):
            mapper = self.model.output_head.distribution_mappers[var_name]
            dist_support = mapper.support
            if not transformed_support.is_subset(dist_support, idx=i):
                dist_name = self.model.output_head.variable_to_distribution[var_name]
                interval = self._format_support_interval(transformed_support, idx=i)
                support_name = self._format_distribution_support(dist_support)
                raise ValueError(
                    "Output support validation failed for variable "
                    f"'{var_name}' (index={i}, distribution='{dist_name}'). "
                    f"Distribution support is {support_name}, but transformed target "
                    f"support is {interval}. Ensure output_transform and "
                    "obs_variable_to_distribution are compatible."
                )

    def on_fit_start(self) -> None:
        train_loader = self.trainer.train_dataloader
        train_dataset = train_loader.dataset if hasattr(train_loader, "dataset") else None

        if train_dataset is not None:
            self.validate_output_head_supports(train_dataset)
        
        # If there is a OAT loss scheduler, 
        # check that the total steps given to the scheduler matches the trainer's estimated total steps. 
        # If not, log a warning.
        if self.oat_loss_weight_scheduler is not None:

            total_steps = self.trainer.estimated_stepping_batches 
            if total_steps is None:
                if self.global_rank == 0:
                    logger.warning(
                        "WARNING: Could not infer trainer estimated stepping batches; "
                        "skipping OAT scheduler total-step consistency check."
                    )
                return
        
            requested_total_step = self.oat_loss_weight_scheduler.total_steps
            
            if requested_total_step != total_steps:
                if self.global_rank == 0:
                    logger.warning(
                        f"WARNING: Scheduler configured for {requested_total_step} steps "
                        f"but trainer estimates {total_steps} steps."
                        " Ensure this is voluntary or set the steps to null in your config"
                )
            else:
                if self.global_rank == 0:
                    logger.info(f"OAT loss scheduler total steps matches trainer's estimated steps: {total_steps}.")


    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        # This is done in parallel by all group members.
        # Ie. all GPUs advance their copy of the scheduler by one step
        if self.oat_loss_weight_scheduler is not None:
            oat_weight = self.oat_loss_weight_scheduler.step()
            self.model.pre_oat_loss_weight = oat_weight
            
            self.log("pre_oat_loss_weight", oat_weight, prog_bar=False, on_step=True, on_epoch=False)

    def _normalize_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.input_transform(batch)

    def _unnormalize_pred_and_target(self, y_hat: dict[str, torch.Tensor], y_true: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Given the model's prediction and normalized targets, de-normalizes the predicted samples
        and and target and return them.

        Args:
            y_hat: the model's prediction containing samples
            y_true: the normalized target

        Returns:
           The de-normalized samples and de-normalized targets
        """
        return self.output_transform.inverse(y_hat['samples']), self.output_transform.inverse(y_true)

    def forward(self, batch: dict[str, torch.Tensor], query_locations: Optional[torch.Tensor] = None, return_loss_and_targets: bool = False, n_samples: Optional[int] = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the OATPredictor.
        It expects the input data to be normalized. The user should prefer using
        training_step, validation_step, test_step or predict_step over this method
        directly.

        Args:
            batch : The input data batch
            query_locations : Location of query stations. When not provided, the input batch is expected
            to already have a key "query_data". This parameter should be used during inference.
            return_loss_and_targets: Whether loss, target and target mask should be returned.

        Returns:
            The prediction as well as loss, target and target mask when return_loss_and_targets is True.
        """

        if query_locations is None:
            # If no query locations are given, we assume that they are given by the dataloader
            assert "query_data" in batch.keys(), "Batch does not contain query data and no query\
                locations have been provided"

            target_obs = batch['query_data'].pop('obs', None)
            target_mask_obs = batch['query_data'].pop('mask_obs', None)
        else:
            # TODO implement 
            pass
        
        # Prepare arguments:
        # there are arguments common to both informed and uninformed models
        model_fwd_kwargs = dict(
            global_data=batch["global_data"],
            query_data=batch["query_data"],
        )
        if self.model.requires_observation:
            # these are arguments for the informed models only
            model_fwd_kwargs["observation_data"] = batch['observation_data']
            model_fwd_kwargs["return_pre_oat_output"] =True

        original_n_samples = None
        if n_samples is not None:
            original_n_samples = self.model.num_samples
            self.model.num_samples = n_samples
        try:
            if not return_loss_and_targets:
                out = self.model.forward(**model_fwd_kwargs)
                return out

            assert query_locations is None, "Can not compute loss for unlabelled queries."
            # thee are arguments only required if you need to return the loss
            model_fwd_kwargs["target_obs"] = target_obs
            model_fwd_kwargs["target_mask_obs"] = target_mask_obs
            out, per_var_losses, loss = self.model.forward_with_loss(**model_fwd_kwargs)

            return out, per_var_losses, loss, target_obs, target_mask_obs
        finally:
            if original_n_samples is not None:
                self.model.num_samples = original_n_samples

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        batch = self._normalize_batch(batch)
        y_hat, per_var_losses, loss, target_obs, target_mask_obs = self(batch, return_loss_and_targets=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self._log_per_var_losses(per_var_losses=per_var_losses,
                                 prefix="train", 
                                 on_step=True, 
                                 on_epoch=True)

        with torch.no_grad():

            # y_hat_unorm shape: [batch,station,sample,feature]
            # target_obs shape: [batch,station,feature]
            # NOTE should we add nograd here?
            y_hat_samples_unnorm, y_true_unnorm = self._unnormalize_pred_and_target(y_hat=y_hat, 
                                                                    y_true=target_obs)

            self._log_instant_metrics(y_hat_samples=y_hat_samples_unnorm,
                                    y_true=y_true_unnorm,
                                    mask=target_mask_obs,
                                    prefix="train",
                                    on_step=True, on_epoch=True)
            
            # pre_oat logging
            if 'pre_oat_output' in y_hat:
                y_hat_samples_unnorm_pre_oat, _ = self._unnormalize_pred_and_target(y_hat=y_hat['pre_oat_output'],
                                                                y_true=target_obs)
                self._log_instant_metrics(y_hat_samples=y_hat_samples_unnorm_pre_oat,
                                        y_true=y_true_unnorm,
                                        mask=target_mask_obs,
                                        prefix="train_pre_oat",
                                        on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        batch = self._normalize_batch(batch)
        y_hat, per_var_losses, loss, target_obs, target_mask_obs = self(batch, return_loss_and_targets=True)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._log_per_var_losses(per_var_losses=per_var_losses,
                                 prefix="val", 
                                 on_step=False, on_epoch=True)
            
        with torch.no_grad():
            y_hat_samples_unnorm, y_true_unnorm = self._unnormalize_pred_and_target(y_hat=y_hat,
                                                            y_true=target_obs)
            # TODO log metrics with unorm value
            self._log_instant_metrics(y_hat_samples=y_hat_samples_unnorm,
                                    y_true=y_true_unnorm,
                                    mask=target_mask_obs,
                                    prefix="val",
                                    on_step=False, on_epoch=True)

            # pre_oat logging
            if 'pre_oat_output' in y_hat:
                    y_hat_samples_unnorm_pre_oat, _ = self._unnormalize_pred_and_target(y_hat=y_hat['pre_oat_output'],
                                                                    y_true=target_obs)
                    self._log_instant_metrics(y_hat_samples=y_hat_samples_unnorm_pre_oat,
                                        y_true=y_true_unnorm,
                                        mask=target_mask_obs,
                                        prefix="val_pre_oat",
                                        on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        batch = self._normalize_batch(batch)
        y_hat, per_var_losses, loss, target_obs, target_mask_obs = self(batch, return_loss_and_targets=True)

        self.log("test_loss", loss)
        self._log_per_var_losses(per_var_losses=per_var_losses,
                                 prefix="test", 
                                 on_step=False, on_epoch=True)
        
        with torch.no_grad():
            
            y_hat_samples_unnorm, y_true_unnorm = self._unnormalize_pred_and_target(y_hat=y_hat,
                                                            y_true=target_obs)

            self._log_instant_metrics(y_hat_samples=y_hat_samples_unnorm,
                                    y_true=y_true_unnorm,
                                    mask=target_mask_obs,
                                    prefix="test",
                                    on_step=False, on_epoch=True)
            
            # pre_oat logging
            if 'pre_oat_output' in y_hat:
                    y_hat_samples_unnorm_pre_oat, _ = self._unnormalize_pred_and_target(y_hat=y_hat['pre_oat_output'],
                                                                    y_true=target_obs)
                    self._log_instant_metrics(y_hat_samples=y_hat_samples_unnorm_pre_oat,
                                        y_true=y_true_unnorm,
                                        mask=target_mask_obs,
                                        prefix="test_pre_oat",
                                        on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0, return_as: Literal['samples', 'distribution'] = 'samples', n_samples: Optional[int] = None) -> torch.Tensor:
        batch = self._normalize_batch(batch)
        if return_as == 'samples':
            with torch.no_grad():
                y_hat = self(batch, return_loss_and_targets=False, n_samples=n_samples)
            return self.output_transform.inverse(y_hat['samples'])
        elif return_as == 'distribution':
            y_hat = self(batch, return_loss_and_targets=False)
            y_hat.pop('samples')
            y_hat.pop('pre_oat_output', None)
            logger.warning("The returned distributions are still parameterized in the transformed space. Ensure to apply the appropriate inverse transform to the distribution parameters before using them.")
            return y_hat
        else:
            raise ValueError(f"Invalid value for return_as: {return_as}. Expected 'samples' or 'distribution'.")
    
    def _log_per_var_losses(self, per_var_losses: dict, prefix: str, on_step: bool, on_epoch: bool) -> None:
        """
        Given a dictionary mapping var_name->loss, logs this as separate metrics.
        This is useful to decide whether a different scaling must be applied to each variable in the 
        loss computation.
        """
        for var_name, loss_val in per_var_losses.items():
            self.log(f'{prefix}_loss_{var_name}', loss_val,
                     prog_bar=False, on_step=on_step, on_epoch=on_epoch)

    def _log_instant_metrics(self, y_hat_samples: torch.Tensor, 
                                        y_true: torch.Tensor, 
                                        mask: torch.Tensor,
                                        prefix: str = "",
                                        on_step: bool = True,
                                        on_epoch: bool = True) -> None:
        for metric_name, metric_fn in self.metrics_fn.items():
            
            metric_values = metric_fn(y_target=y_true,
                                      y_hat=y_hat_samples,
                                      mask=mask,
                                      aggregation='variable')
            
            for i, metric_val in enumerate(metric_values):
                var_name = self.indx_to_var[i]
                
                self.log(name=f'{prefix}_{metric_name}_{var_name}', 
                         value=metric_val, 
                         prog_bar=True, on_step=on_step, on_epoch=on_epoch)
            

    def configure_optimizers(self) -> Optimizer:

            optimizer = self.optimizer_class(self.parameters(), **self.optimizer_params)

            if self.lr_scheduler_class is not None:
                scheduler = self.lr_scheduler_class(optimizer, **self.lr_scheduler_params)
                lr_scheduler_cfg = {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                    "name": 'lr_scheduler' # for `LearningRateMonitor` callback
                }

                return {
                    "optimizer": optimizer,
                    "lr_scheduler": lr_scheduler_cfg
                }
            return optimizer


def setup_oat_cfg(cfg: DictConfig, field_extent: Dict, logger: logging.Logger) -> model.OATConfig | model.UninformedOATConfig:
    model_cfg = cfg.model
    data_cfg = cfg.dataset
    logger.info("Setting up model configs...")
    global_input_chans = len(data_cfg.nwp_vars) + len(data_cfg.topo_vars)
    global_encoder_cfg = model.GlobalEncoderConfig(
        **model_cfg.global_encoder,
        in_channels=global_input_chans,
        field_extent_max=field_extent['max'],
        field_extent_min=field_extent['min']
        )

    topography_encoder_cfg = model.TopographicEncoderConfig(
        **model_cfg.topographic_encoder,
        in_channels=len(data_cfg.topo_vars)
        )

    time_cond_dim = 4 if data_cfg.use_time_embedding else 0
    nwp_topo_fusion_cfg = model.NwpTopoFusionConfig(
        **model_cfg.nwp_topo_fusion,
        cond_dim=time_cond_dim,
        out_channels=model_cfg.d_model)

    if list(cfg.dataset.obs_variable_to_distribution.items()) != list(cfg.model.output_head.variable_to_distribution.items()):
        raise ValueError("Mismatch between observations variable distributions and output head variable distributions (both keys and order must match)")
    if list(cfg.model.output_head.variable_to_distribution.keys()) != cfg.dataset.obs_vars:
        raise ValueError("Variables in output head distribution config do not match observation variables")
    if len(cfg.model.output_head.variable_to_distribution) != len(cfg.dataset.obs_vars):
        raise ValueError("Number of variables in output head distribution config does not match number of observation variables")
    logger.info(
        f"Model will assume the following index to variable mapping: "
        f"{ {idx: var for idx, var in enumerate(cfg.dataset.obs_vars)} }"
        "\n And variable to distribution mapping: "
        f"{ cfg.model.output_head.variable_to_distribution }"
    )
    output_head_cfg = model.OutputHeadConfig(
        in_features=model_cfg.d_model,
        **model_cfg.output_head)

    if get_model_cls(cfg) == model.UninformedOAT:
        logger.info("Initialized uninformed model cfg...")
        return model.UninformedOATConfig(
            global_encoder=global_encoder_cfg,
            topographic_encoder=topography_encoder_cfg,
            nwp_topo_fusion=nwp_topo_fusion_cfg,
            output_head=output_head_cfg,
            num_samples=model_cfg.num_samples
        )

    masking_cfg = model.MaskingConfig(**model_cfg.masking)

    transformer_cfg = model.TransformerConfig(
        d_geo_bias=len(data_cfg.geo_vars),
        d_model=model_cfg.d_model,
        **model_cfg.transformer)

    n_obs_vars = len(data_cfg.obs_vars)
    condition_on_mask = model_cfg.innovation_fusion.condition_on_mask
    innovation_cond_dim = [time_cond_dim, n_obs_vars] if condition_on_mask else time_cond_dim
    innovation_fusion_cfg = model.InnovationFusionConfig(
        **model_cfg.innovation_fusion,
        cond_dim=innovation_cond_dim,
        out_channels=model_cfg.d_model)

    if not model_cfg.get("learnable_gating", False):
        logger.info("Learnable gating disabled.")
        learnable_gating_cfg = None
    else:
        logger.info("Learnable gating enabled.")
        learnable_gating_cfg = model.LearnableGatingConfig(
            **model_cfg.learnable_gating,
            in_channels=n_obs_vars,
            out_channels=model_cfg.d_model)

    oat_config = model.OATConfig(
        global_encoder=global_encoder_cfg,
        topographic_encoder=topography_encoder_cfg,
        nwp_topo_fusion=nwp_topo_fusion_cfg,
        innovation_fusion=innovation_fusion_cfg,
        masking=masking_cfg,
        transformer=transformer_cfg,
        output_head=output_head_cfg,
        learnable_gating=learnable_gating_cfg,
        num_samples=model_cfg.num_samples,
        pre_oat_loss_weight=model_cfg.pre_oat_loss_weight,
        mask_innovations=model_cfg.mask_innovations
    )
    logger.info("Initialized model cfg...")
    return oat_config


def setup_oat_predictor(model_cfg: model.OATConfig | model.UninformedOATConfig,
                    estimated_total_step: int,
                    var_names: list[str],
                    transforms: tuple[Transform, Transform], cfg: DictConfig) -> OATPredictor:

    # optimizer
    module_path, class_name = cfg.predictor.kwargs.optimizer_class.rsplit(".", 1)
    module = import_module(module_path)
    optim_cls = getattr(module, class_name)

    # lr scheduler
    module_path, class_name = cfg.predictor.kwargs.lr_scheduler_class.rsplit(".", 1)
    module = import_module(module_path)
    sched_cls = getattr(module, class_name)

    # oat loss scheduler
    oat_loss_scheduler_class = cfg.predictor.kwargs.get("oat_loss_scheduler_class", None)
    if oat_loss_scheduler_class is not None:
        module_path, class_name = cfg.predictor.kwargs.oat_loss_scheduler_class.rsplit(".", 1)
        module = import_module(module_path)
        oat_loss_sched_cls = getattr(module, class_name)
        oat_loss_sched_params = cfg.predictor.kwargs.oat_loss_scheduler_params
        if oat_loss_sched_params['total_steps'] is None:
            oat_loss_sched_params['total_steps'] = estimated_total_step

            logger.info(f"Config did not specify total_steps for the OAT loss scheduler. It has been set to the (manual) estimated total training steps: {estimated_total_step}\
                \nCheck later logs for a comparison with the steps estimated by the lightning trainer.")
        else:
            logger.info(f"OAT loss scheduler total_steps from config: {oat_loss_sched_params['total_steps']} \
                \nCheck later logs for a comparison with the steps estimated by the lightning trainer.")

    else:
        oat_loss_sched_cls = None
        oat_loss_sched_params = None


    predictor = OATPredictor(
        oat_model_cfg=model_cfg,
        input_transform=transforms[0],
        output_transform=transforms[1],
        output_var_names=var_names,
        lr=cfg.predictor.lr,
        optimizer_class=optim_cls,
        optimizer_params=cfg.predictor.kwargs.optimizer_params,
        lr_scheduler_class=sched_cls,
        lr_scheduler_params=cfg.predictor.kwargs.lr_scheduler_params,
        oat_loss_scheduler_class=oat_loss_sched_cls,
        oat_loss_scheduler_params=oat_loss_sched_params
    )

    return predictor
