from __future__ import annotations

from typing import Any, List, Optional, Type

import pytorch_lightning as pl
import torch
from torch.optim.optimizer import Optimizer

from oat.data_module.transform import Transform
from oat.model_module.gp_baseline import GPBaseline
from oat.predictor_module.functional_metrics import mc_crps, point_mae


class GPPredictor(pl.LightningModule):
    def __init__(
        self,
        gp_model: GPBaseline,
        input_transform: Transform,
        output_transform: Transform,
        output_var_names: List[str],
        lr: float = 1e-3,
        optimizer_class: Type[Optimizer] = torch.optim.Adam,
        optimizer_params: Optional[dict] = None,
        scheduler_class: Optional[Type] = None,
        scheduler_params: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.model = gp_model
        self.input_transform = input_transform
        self.output_transform = output_transform

        self.output_var_names = output_var_names
        self.var_to_indx = {v: i for i, v in enumerate(output_var_names)}
        self.indx_to_var = {i: v for i, v in enumerate(output_var_names)}

        self.optimizer_class = optimizer_class
        self.optimizer_params = {"lr": lr, **(optimizer_params or {})}
        self.scheduler_class = scheduler_class
        self.scheduler_params = scheduler_params or {}

        self.metrics_fn = {
            "point_mae": point_mae,
            "crps": mc_crps,
        }

    def _normalize_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self.input_transform(batch)

    def _unnormalize_pred_and_target(
        self,
        y_hat: dict[str, torch.Tensor],
        y_true: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.output_transform.inverse(y_hat["samples"]),
            self.output_transform.inverse(y_true),
        )

    def _log_instant_metrics(
        self,
        y_hat_samples: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
        prefix: str = "",
        on_step: bool = True,
        on_epoch: bool = True,
    ) -> None:
        for metric_name, metric_fn in self.metrics_fn.items():
            metric_values = metric_fn(
                y_target=y_true,
                y_hat=y_hat_samples,
                mask=mask,
                aggregation="variable",
            )
            for i, metric_val in enumerate(metric_values):
                self.log(
                    name=f"{prefix}_{metric_name}_{self.indx_to_var[i]}",
                    value=metric_val,
                    prog_bar=True,
                    on_step=on_step,
                    on_epoch=on_epoch,
                )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch = self._normalize_batch(batch)
        target_obs = batch["query_data"].pop("obs")
        target_mask_obs = batch["query_data"].pop("mask_obs")

        y_hat, loss = self.model.forward_with_loss(
            global_data=batch["global_data"],
            observation_data=batch["observation_data"],
            query_data=batch["query_data"],
            target_obs=target_obs,
            target_mask_obs=target_mask_obs,
        )
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        with torch.no_grad():
            y_hat_unnorm, y_true_unnorm = self._unnormalize_pred_and_target(y_hat, target_obs)
            self._log_instant_metrics(
                y_hat_samples=y_hat_unnorm,
                y_true=y_true_unnorm,
                mask=target_mask_obs,
                prefix="train",
                on_step=True,
                on_epoch=True,
            )

        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch = self._normalize_batch(batch)
        target_obs = batch["query_data"].pop("obs")
        target_mask_obs = batch["query_data"].pop("mask_obs")

        y_hat, loss = self.model.forward_with_loss(
            global_data=batch["global_data"],
            observation_data=batch["observation_data"],
            query_data=batch["query_data"],
            target_obs=target_obs,
            target_mask_obs=target_mask_obs,
        )
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        with torch.no_grad():
            y_hat_unnorm, y_true_unnorm = self._unnormalize_pred_and_target(y_hat, target_obs)
            self._log_instant_metrics(
                y_hat_samples=y_hat_unnorm,
                y_true=y_true_unnorm,
                mask=target_mask_obs,
                prefix="val",
                on_step=False,
                on_epoch=True,
            )

        return loss

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch = self._normalize_batch(batch)
        target_obs = batch["query_data"].pop("obs")
        target_mask_obs = batch["query_data"].pop("mask_obs")

        y_hat, loss = self.model.forward_with_loss(
            global_data=batch["global_data"],
            observation_data=batch["observation_data"],
            query_data=batch["query_data"],
            target_obs=target_obs,
            target_mask_obs=target_mask_obs,
        )
        self.log("test_loss", loss)

        with torch.no_grad():
            y_hat_unnorm, y_true_unnorm = self._unnormalize_pred_and_target(y_hat, target_obs)
            self._log_instant_metrics(
                y_hat_samples=y_hat_unnorm,
                y_true=y_true_unnorm,
                mask=target_mask_obs,
                prefix="test",
                on_step=False,
                on_epoch=True,
            )

        return loss

    def predict_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        batch = self._normalize_batch(batch)
        y_hat = self.model.forward(
            global_data=batch["global_data"],
            observation_data=batch["observation_data"],
            query_data=batch["query_data"],
        )
        return self.output_transform.inverse(y_hat["samples"])

    def configure_optimizers(self) -> Optimizer | dict:
        optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_params)
        if self.scheduler_class is not None:
            scheduler = self.scheduler_class(optimizer, **self.scheduler_params)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                    "name": "lr_scheduler",
                },
            }
        return optimizer
