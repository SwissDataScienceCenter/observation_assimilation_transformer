from typing import Type, Optional
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from oat.data_module.transform import Transform

class OATPredictor(pl.LightningModule):

    def __init__(self,
                 oat_model: nn.Module,
                 input_transform: Transform,
                 output_transform: Transform,
                 lr: float = 1e-3,
                 training_stations_split: float = 0.2,
                 optimizer_class: Type[Optimizer] = torch.optim.Adam,
                 optimizer_params: Optional[dict] = None,
                 scheduler_class: Type[LRScheduler] | None = None,
                 scheduler_params: Optional[dict] = None
                 ):
        super().__init__()

        # Ignore the model itself in hparams but update manually with its cfg
        self.save_hyperparameters(ignore=['oat_model', 'input_transform', 'output_transform'])
        self.hparams.update({'model_cfg': oat_model.config})

        self.training_stations_split = training_stations_split
        self.optimizer_class = optimizer_class
        self.optimizer_params = {"lr": lr, **(optimizer_params or {})}

        self.scheduler_class = scheduler_class
        self.scheduler_params = scheduler_params or {}

        self.model = oat_model
        self.input_transform = input_transform
        self.output_transform = output_transform

    def _normalize_batch(self, batch):
        # TODO implement
        return self.input_transform(batch)

    def _unnormalize_prediction(self, y_hat: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Given the model's prediction, de-normalizes the predicted samples
        and return them.

        Args:
            y_hat: the model's prediction containing samples

        Returns:
           The tensor with de-normalized samples.
        """
        return self.output_transform(y_hat['samples'])

    def forward(self, batch, query_locations = None, return_loss_and_targets = False):

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
                               query_data=batch['query_data'])

            return out
        else:
            assert query_locations is None, "Can not compute loss for unlabelled queries."
            out, loss = self.model.forward_with_loss(global_data=batch['global_data'],
                                                     observation_data=batch['observation_data'],
                                                     query_data=batch['query_data'],
                                                     target_obs=target_obs,
                                                     target_mask_obs=target_mask_obs)

            return out, loss, target_obs, target_mask_obs

    def training_step(self, batch, batch_idx):
        batch = self._normalize_batch(batch)
        y_hat, loss, target_obs, target_mask_obs= self(batch, return_loss_and_targets=True)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # y_hat_unorm shape: [batch,station,sample,feature]
        # target_obs shape: [batch,station,feature]
        y_hat_unorm = self._unnormalize_prediction(y_hat)
        # TODO targets are to be unormalized too
        
        # TODO log metrics with unorm value


        return loss

    def validation_step(self, batch, batch_idx):
        batch = self._normalize_batch(batch)
        y_hat, loss, target_obs, target_mask_obs = self(batch, return_loss_and_targets=True)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        y_hat_unorm = self._unnormalize_prediction(y_hat)
        # TODO log metrics with unorm value

        return loss

    def test_step(self, batch, batch_idx):
        batch = self._normalize_batch(batch)
        y_hat, loss, target_obs, target_mask_obs = self(batch, return_loss_and_targets=True)

        self.log("test_loss", loss)

        y_hat_unorm = self._unnormalize_prediction(y_hat)
        # TODO log metrics with unorm value

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batch = self._normalize_batch(batch)

        y_hat = self(batch, return_loss=False)

        return self._unnormalize_prediction(y_hat)

    def configure_optimizers(self):

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
