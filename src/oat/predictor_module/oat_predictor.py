from typing import Type, Optional, List
import logging

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from oat.model_module.model import (
    OATConfig, 
    OAT, 
    UninformedOATConfig,
    UninformedOAT
)
from oat.data_module.transform import Transform
from oat.predictor_module.functional_metrics import (
    sample_mae, 
    point_mae,
    mc_crps
)

logger = logging.getLogger(__name__)

class OATPredictor(pl.LightningModule):

    def __init__(self,
                 oat_model_cfg: OATConfig | UninformedOATConfig,
                 input_transform: Transform,
                 output_transform: Transform,
                 output_var_names: List,
                 lr: float = 1e-3,
                 optimizer_class: Type[Optimizer] = torch.optim.Adam,
                 optimizer_params: Optional[dict] = None,
                 scheduler_class: Type[LRScheduler] | None = None,
                 scheduler_params: Optional[dict] = None,
                 oat_loss_scheduler_cls: Optional[dict] = None,
                 oat_loss_scheduler_params:  Optional[dict] = None,
                 ):
        super().__init__()

        self.save_hyperparameters()

        self.optimizer_class = optimizer_class
        self.optimizer_params = {"lr": lr, **(optimizer_params or {})}

        self.scheduler_class = scheduler_class
        self.scheduler_params = scheduler_params or {}

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
        
        if oat_loss_scheduler_cls is not None:
            assert oat_loss_scheduler_params is not None
            assert isinstance(oat_model_cfg, OATConfig)
            # self.oat_loss_weight_scheduler = oat_loss_scheduler_cls(**oat_loss_scheduler_params)
            self.oat_loss_scheduler_cls = oat_loss_scheduler_cls
            self.oat_loss_scheduler_params = oat_loss_scheduler_params
            logger.info(f"OAT loss scheduler {oat_loss_scheduler_cls} will be lazily initialized when fitting starts.")
        else:
            self.oat_loss_weight_scheduler = None

    def on_fit_start(self):
        
        # Case 1; there is no pre-oat scheduler
        if self.oat_loss_scheduler_cls is None:
            return

        # Case 2; the scheduler was already initialized, this is probably a checkpoint
        # where the state_dict has been already loaded.
        if self.oat_loss_weight_scheduler is not None:
            logger.info("OAT loss scheduler already exists. Skipping initialization.")
            return

        # Case 3: logger is not yet initialize (lazy)

        # this is only available when fitting starts
        total_steps = self.trainer.estimated_stepping_batches 
        
        if self.oat_loss_scheduler_params.get("total_steps") is not None:
            requested_steps = self.oat_loss_scheduler_params["total_steps"]
            if requested_steps != total_steps:
                if self.global_rank == 0:
                    logger.info(
                        f"WARNING: Scheduler configured for {requested_steps} steps "
                        f"but trainer estimates {total_steps} steps."
                        "Ensure this is is voluntary or set the steps to null in your config"
                )
        else:
            self.oat_loss_scheduler_params["total_steps"] = total_steps
            
        self.oat_loss_weight_scheduler = self.oat_loss_scheduler_cls(**self.oat_loss_scheduler_params)

        if self.global_rank == 0:
            logger.info("OAT loss scheduler initialized.")
            logger.info(self.oat_loss_weight_scheduler) # print printable representation of the scheduler


    def on_train_batch_end(self, *args, **kwargs):
        # This is done in parallel by all group members.
        # Ie. all GPUs advance their copy of the scheduler by one step
        if self.oat_loss_weight_scheduler is not None:
            oat_weight = self.oat_loss_weight_scheduler.step()
            self.model.set_pre_oat_loss_weight(oat_weight)
            
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

            if self.scheduler_class is not None:
                scheduler = self.scheduler_class(optimizer, **self.scheduler_params)
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
