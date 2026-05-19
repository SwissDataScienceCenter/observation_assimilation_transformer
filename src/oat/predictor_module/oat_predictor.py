from typing import Any, Type, Optional, List
import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from oat.model_module.model import (
    OATConfig, 
    OAT, 
    UninformedOATConfig,
    UninformedOAT
)
from oat.data_module.transform import Transform
from oat.data_module.support import VariableSupport
from oat.predictor_module.functional_metrics import (
    sample_mae, 
    point_mae,
    mc_crps
)

logger = logging.getLogger(__name__)

class OATPredictor(pl.LightningModule):
    
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
        super().__init__()

        self.save_hyperparameters()

        self.optimizer_class = optimizer_class
        self.optimizer_params = {"lr": lr, **(optimizer_params or {})}

        self.lr_scheduler_class = lr_scheduler_class
        self.lr_scheduler_params = lr_scheduler_params or {}

        if isinstance(oat_model_cfg, OATConfig):
            self.model = OAT(oat_model_cfg)
        elif isinstance(oat_model_cfg, UninformedOATConfig):
            self.model = UninformedOAT(oat_model_cfg)

        self.input_transform = input_transform
        self.output_transform = output_transform
        
        self.output_var_names = output_var_names
        self.var_to_indx = {v: i for i,v in enumerate(self.output_var_names)}
        self.indx_to_var = {i: v for i,v in enumerate(self.output_var_names)}
        
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

        assert list(self.model.output_head.variable_to_distribution.keys()) == self.output_var_names

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

    def forward(self, batch: dict[str, torch.Tensor], query_locations: Optional[torch.Tensor] = None, return_loss_and_targets: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        if not return_loss_and_targets:
            out = self.model.forward(**model_fwd_kwargs)
            return out

        assert query_locations is None, "Can not compute loss for unlabelled queries."
        # thee are arguments only required if you need to return the loss
        model_fwd_kwargs["target_obs"] = target_obs
        model_fwd_kwargs["target_mask_obs"] = target_mask_obs
        out, loss = self.model.forward_with_loss(**model_fwd_kwargs)

        return out, loss, target_obs, target_mask_obs

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        batch = self._normalize_batch(batch)
        y_hat, loss, target_obs, target_mask_obs= self(batch, return_loss_and_targets=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

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
        y_hat, loss, target_obs, target_mask_obs = self(batch, return_loss_and_targets=True)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            
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
        y_hat, loss, target_obs, target_mask_obs = self(batch, return_loss_and_targets=True)

        self.log("test_loss", loss)
        
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

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        batch = self._normalize_batch(batch)

        y_hat = self(batch, return_loss_and_targets=False)

        return self.output_transform.inverse(y_hat['samples'])
    
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
