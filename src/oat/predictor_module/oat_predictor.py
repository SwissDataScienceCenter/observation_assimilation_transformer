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
                 optimizer_class: Type[Optimizer] = torch.optim.Adam,
                 optimizer_params: Optional[dict] = None,
                 scheduler_class: Type[LRScheduler] | None = None,
                 scheduler_params: Optional[dict] = None
                 ):
        super().__init__()

        self.save_hyperparameters(ignore=['oat_model'])

        self.optimizer_class = optimizer_class
        self.optimizer_params = {"lr": lr, **(optimizer_params or {})}

        self.scheduler_class = scheduler_class
        self.scheduler_params = scheduler_params or {}

    def _normalize_batch(self, batch):
        # TODO implement
        pass

    def _unnormalize_prediction(self, pred):
        # TODO implement
        pass

    def forward(self, x, return_loss = False):
        # TODO implement
        pass

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
