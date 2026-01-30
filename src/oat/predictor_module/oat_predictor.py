from typing import Type, Optional
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

class OATPredictor(pl.LightningModule):

    def __init__(self,
                 oat_model: nn.Module,
                 lr: float = 1e-3,
                 training_stations_split: float = 0.2,
                 optimizer_class: Type[Optimizer] = torch.optim.Adam,
                 optimizer_params: Optional[dict] = None,
                 scheduler_class: Type[LRScheduler] | None = None,
                 scheduler_params: Optional[dict] = None
                 ):
        super().__init__()
        # TODO here if we pass the model, we have to ignore it from the hyperparams
        # (otherwise it's just serialized weights). However it would be nice to save 
        # the architecture of the model.
        # - Can we save the cfg that generated the model? 
        # - Should we pass the cfg, and the model is initialized in here?
        self.save_hyperparameters(ignore=['oat_model'])

        self.training_stations_split = training_stations_split
        self.optimizer_class = optimizer_class
        self.optimizer_params = {"lr": lr, **(optimizer_params or {})}

        self.scheduler_class = scheduler_class
        self.scheduler_params = scheduler_params or {}

        self.model = oat_model

    def _normalize_batch(self, batch):
        # TODO implement
        pass

    def _unnormalize_prediction(self, pred):
        # TODO implement
        pass


    def _split_queries_key_values(self, batch):
        assert batch['observation_data']['obs'] is not None

        n_stations = batch['observation_data']['coords'].shape[1]
        n_select = int(self.training_stations_split * n_stations)

        q_station_idx = torch.randperm(n_stations)[:n_select]
        kv_station_idx = torch.randperm(n_stations)[n_select:]

        all_data = batch['observation_data']
        # select key-value stations
        batch['observation_data'] = {
                k: v.index_select(1, kv_station_idx) if v.shape[1]  == n_stations else v
                for k,v in all_data.items()
        }
        # select query stations and drop observational keys (which will be targets)
        batch['query_data'] = {
                k: v.index_select(1, q_station_idx) if v.shape[1]  == n_stations else v
                for k,v in all_data.items() 
        }

        target_obs = batch['query_data'].pop('obs', None)
        target_mask_obs = batch['query_data'].pop('mask_obs', None)

        return batch, target_obs, target_mask_obs

    def forward(self, batch, query_locations = None, return_loss = False):

        if query_locations is None:
            # If no query locations are given, a subset of the batch is used as query
            batch, target_obs, target_mask_obs = self._split_queries_key_values(batch=batch)
        else:
            # TODO implement
            pass

        if not return_loss:
            out = self.model.forward(global_data=batch['global_data'],
                               observation_data=batch['observation_data'],
                               query_data=batch['query_data'])

            return out
        else:
            assert query_locations is None, "Can not compute loss for unlabelled queries."
            out, loss = self.model.forward_with_loss(global_data=batch['global_data'],
                                                     observation_data=batch['observation_data'],
                                                     query_data=batch['query_data'],
                                                     target_obs=target_mask_obs,
                                                     target_mask_obs=target_mask_obs)

            return out, loss

    def training_step(self, batch, batch_idx):
        batch = self._normalize_batch(batch)
        y_hat, loss = self(batch, return_loss=True)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        y_hat_unorm = self._unnormalize_prediction(y_hat)
        # TODO log metrics with unorm value

        return loss

    def validation_step(self, batch, batch_idx):
        batch = self._normalize_batch(batch)
        y_hat, loss = self(batch, return_loss=True)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        y_hat_unorm = self._unnormalize_prediction(y_hat)
        # TODO log metrics with unorm value

        return loss

    def test_step(self, batch, batch_idx):
        batch = self._normalize_batch(batch)
        y_hat, loss = self(batch, return_loss=True)

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
