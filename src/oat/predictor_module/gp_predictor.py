from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type
from importlib import import_module

import pytorch_lightning as pl
import torch
from torch.optim.optimizer import Optimizer
from omegaconf import DictConfig

from oat.data_module.transform import Transform
from oat.model_module.gp_baseline import GPBaseline, GPBaselineConfig
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


def setup_gp_predictor(
    cfg: DictConfig,
    field_extent: Dict,
    obs_vars: List[str],
    transforms: Tuple[Transform, Transform],
) -> "GPPredictor":
    data_cfg = cfg.dataset
    model_cfg = cfg.model

    gp_topography_kwargs = None
    if bool(model_cfg.get("use_topography", False)):
        gp_topography_kwargs = {
            **dict(model_cfg.topographic_encoder),
            "in_channels": len(data_cfg.topo_vars),
        }

    gp_config = GPBaselineConfig(
        obs_vars=obs_vars,
        nwp_vars=list(data_cfg.nwp_vars),
        field_extent=field_extent,
        use_nwp=bool(model_cfg.use_nwp),
        use_time_emb=bool(model_cfg.use_time_emb),
        use_topography=bool(model_cfg.get("use_topography", False)),
        topography_encoder_kwargs=gp_topography_kwargs,
        mean_hidden_dims=tuple(model_cfg.mean_hidden_dims),
        kernel_hidden_dims=tuple(model_cfg.kernel_hidden_dims),
        kernel_embed_dim=int(model_cfg.kernel_embed_dim),
        mean_groups=model_cfg.get("mean_groups"),
        stationary_cov_groups=model_cfg.get("stationary_cov_groups"),
        non_stationary_cov_groups=model_cfg.get("non_stationary_cov_groups"),
        noise_variance=float(model_cfg.noise_variance),
        eps=float(model_cfg.get("eps", 1e-3)),
        loss_fn=str(model_cfg.loss_fn),
        num_samples=int(model_cfg.num_samples),
    )
    gp_model = GPBaseline(gp_config)

    module_path, class_name = cfg.predictor.kwargs.optimizer_class.rsplit(".", 1)
    optim_cls = getattr(import_module(module_path), class_name)

    sched_cls = None
    sched_params: dict = {}
    if cfg.predictor.kwargs.get("scheduler_class") is not None:
        module_path, class_name = cfg.predictor.kwargs.scheduler_class.rsplit(".", 1)
        sched_cls = getattr(import_module(module_path), class_name)
        sched_params = dict(cfg.predictor.kwargs.get("scheduler_params", {}))

    return GPPredictor(
        gp_model=gp_model,
        input_transform=transforms[0],
        output_transform=transforms[1],
        output_var_names=obs_vars,
        lr=float(cfg.predictor.lr),
        optimizer_class=optim_cls,
        optimizer_params=dict(cfg.predictor.kwargs.get("optimizer_params", {})),
        scheduler_class=sched_cls,
        scheduler_params=sched_params,
    )
