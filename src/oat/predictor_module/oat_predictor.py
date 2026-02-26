from typing import Type, Optional, List
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from oat.model_module.model import OATConfig, OAT
from oat.data_module.transform import Transform
from oat.predictor_module.functional_metrics import (
    sample_mae, 
    point_mae,
    mc_crps
)

class OATPredictor(pl.LightningModule):

    def __init__(self,
                 oat_model_cfg: OATConfig,
                 input_transform: Transform,
                 output_transform: Transform,
                 output_var_names: List,
                 lr: float = 1e-3,
                 optimizer_class: Type[Optimizer] = torch.optim.Adam,
                 optimizer_params: Optional[dict] = None,
                 scheduler_class: Type[LRScheduler] | None = None,
                 scheduler_params: Optional[dict] = None
                 ):
        super().__init__()

        # Ignore the model itself in hparams but update manually with its cfg
        self.save_hyperparameters()

        self.optimizer_class = optimizer_class
        self.optimizer_params = {"lr": lr, **(optimizer_params or {})}

        self.scheduler_class = scheduler_class
        self.scheduler_params = scheduler_params or {}

        self.model = OAT(oat_model_cfg)
        self.input_transform = input_transform
        self.output_transform = output_transform
        
        self.output_var_names = output_var_names
        self.var_to_indx = {v: i for i,v in enumerate(self.output_var_names)}
        self.indx_to_var = {i: v for i,v in enumerate(self.output_var_names)}
        
        # NOTE perhaps make this configurable?
        self.metrics_fn = {
            'sample_mae': sample_mae,
            'point_mae': point_mae,
            'crps': mc_crps,
        }

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

        if not return_loss_and_targets:
            out = self.model.forward(global_data=batch['global_data'],
                               observation_data=batch['observation_data'],
                               query_data=batch['query_data'],
                               return_pre_oat_output=True)

            return out

        assert query_locations is None, "Can not compute loss for unlabelled queries."
        out, loss = self.model.forward_with_loss(global_data=batch['global_data'],
                                                    observation_data=batch['observation_data'],
                                                    query_data=batch['query_data'],
                                                    target_obs=target_obs,
                                                    target_mask_obs=target_mask_obs,
                                                    return_pre_oat_output=True)

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
