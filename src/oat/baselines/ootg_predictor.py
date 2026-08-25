"""OOTG baseline predictor for the field-batch data path."""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from oat.data_module.transform import Transform
from oat.predictor_module.base_predictor import BasePredictor
from oat.predictor_module.field_batch_utils import _unwrap_fields_to_ootg_batch
from oat.predictor_module.functional_metrics import mc_crps, point_mae


class OOTGPredictor(BasePredictor):
    """Wraps a trained OOTGGriddedTNP to train and evaluate on field batches.

    Each step unwraps the field-collated batch to an ``OOTGBatch`` via
    ``_unwrap_fields_to_ootg_batch`` and calls the model with positional
    tensors. ``predict_step`` returns samples in physical units via
    ``output_transform.inverse``.

    Outputs [B, N_q, n_samples, n_vars] from predict_step.
    """

    def __init__(
        self,
        model: nn.Module,
        roles: dict[str, str],
        coord_to_latlon: Callable[[torch.Tensor], torch.Tensor],
        grid_shape: tuple[int, int],
        nwp_stride: int,
        output_transform: Transform,
        input_transform: Any,
        output_var_names: list[str],
        lr: float = 5e-4,
        n_samples: int = 20,
        ablate_obs: bool = False,
        ablate_grid: bool = False,
    ) -> None:
        super().__init__(output_var_names=output_var_names, field_roles=roles)
        self.model = model
        self.roles = roles
        self.coord_to_latlon = coord_to_latlon
        self.grid_shape = grid_shape
        self.nwp_stride = nwp_stride
        self.output_transform = output_transform
        self.input_transform = input_transform
        self.lr = lr
        self.n_samples = n_samples
        self.ablate_obs = ablate_obs
        self.ablate_grid = ablate_grid
        self.metrics_fn = {"point_mae": point_mae, "crps": mc_crps}

    def _normalize_batch(self, batch: dict[str, Any]) -> Any:
        """Apply the input transform and unwrap the batch to an OOTGBatch."""
        batch = self.input_transform(batch)
        return _unwrap_fields_to_ootg_batch(
            batch,
            self.roles,
            self.coord_to_latlon,
            self.grid_shape,
            self.nwp_stride,
            self.ablate_obs,
            self.ablate_grid,
        )

    def _forward(self, ootg_batch: Any) -> Any:
        """Run the model on an unwrapped OOTGBatch.

        The model forward takes positional tensors; unpack the batch
        attributes into the call. When the batch carries topo_c/topo_t
        (set by _unwrap_fields_to_ootg_batch when the topo role exists),
        pass them to the TopoOOTGGriddedTNP variant.
        """
        args = (
            ootg_batch.xc,
            ootg_batch.yc,
            ootg_batch.xc_grid,
            ootg_batch.yc_grid,
            ootg_batch.xt,
        )
        if hasattr(ootg_batch, "topo_c"):
            args = (*args, ootg_batch.topo_c, ootg_batch.topo_t)
        return self.model(*args)

    def _draw_samples(self, dist: Any) -> torch.Tensor:
        """Draw self.n_samples samples in normalized space.

        Returns:
            [B, N_q, n_samples, dim_y] tensor
        """
        raw: torch.Tensor = dist.sample(torch.Size([self.n_samples]))
        return raw.permute(1, 2, 0, 3)  # [B, N_q, n, dim_y]

    def _log_instant_metrics(
        self,
        y_hat_samples: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor | None,
        prefix: str = "",
        on_step: bool = True,
        on_epoch: bool = True,
    ) -> None:
        for metric_name, metric_fn in self.metrics_fn.items():
            vals = metric_fn(
                y_target=y_true,
                y_hat=y_hat_samples,
                mask=mask,
                aggregation="variable",
            )
            for i, val in enumerate(vals):
                self.log(
                    f"{prefix}_{metric_name}_{self.indx_to_var[i]}",
                    val,
                    prog_bar=True,
                    on_step=on_step,
                    on_epoch=on_epoch,
                )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        ootg_batch = self._normalize_batch(batch)
        dist = self._forward(ootg_batch)
        loss: torch.Tensor = dist.crps(ootg_batch.yt).mean()
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        query_mask = batch["query_data"]["fields"][self.roles["target"]].mask
        ootg_batch = self._normalize_batch(batch)
        dist = self._forward(ootg_batch)
        loss = dist.crps(ootg_batch.yt).mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        with torch.no_grad():
            samples_phys = self.output_transform.inverse(self._draw_samples(dist))
            yt_phys = self.output_transform.inverse(ootg_batch.yt)
            self._log_instant_metrics(
                samples_phys,
                yt_phys,
                query_mask,
                prefix="val",
                on_step=False,
                on_epoch=True,
            )

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        query_mask = batch["query_data"]["fields"][self.roles["target"]].mask
        ootg_batch = self._normalize_batch(batch)
        dist = self._forward(ootg_batch)
        with torch.no_grad():
            samples_phys = self.output_transform.inverse(self._draw_samples(dist))
            yt_phys = self.output_transform.inverse(ootg_batch.yt)
            self._log_instant_metrics(
                samples_phys,
                yt_phys,
                query_mask,
                prefix="test",
                on_step=False,
                on_epoch=True,
            )

    def predict_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
        n_samples: int | None = None,
    ) -> torch.Tensor:
        ootg_batch = self._normalize_batch(batch)
        dist = self._forward(ootg_batch)
        n = n_samples if n_samples is not None else self.n_samples
        with torch.no_grad():
            raw = dist.sample(torch.Size([n]))
            samples = raw.permute(1, 2, 0, 3)  # [B, N_q, n, dim_y]
        return self.output_transform.inverse(samples)

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)  # type: ignore[attr-defined]


def create_ootg_predictor(
    model: nn.Module,
    dataset: Any,
    output_transform: Transform,
    input_transform: Any,
    cfg: Any,
) -> OOTGPredictor:
    """Build an OOTGPredictor from a field-path dataset and config."""
    from oat.baselines.coord_utils import lonlat_swap, lv95_to_latlon

    ds = getattr(dataset, "dataset", dataset)
    roles = ds.roles()
    dataset_type = ds.config.dataset_type

    coord_fns = {
        "cosmo_peakweather": (lv95_to_latlon, 10),
        "era5_meteostat": (lonlat_swap, 4),
    }
    if dataset_type not in coord_fns:
        raise ValueError(f"Unknown dataset_type for OOTG: {dataset_type!r}")
    coord_to_latlon, default_stride = coord_fns[dataset_type]

    nwp_stride = cfg.get("nwp_stride", None) or default_stride
    grid_field = roles["global_grid"]
    grid_cs = ds.field_coord_sources()[grid_field]
    native_shape = ds.grid_shape(grid_cs)
    if native_shape is None:
        raise ValueError(
            f"No grid shape for coord source {grid_cs!r}; "
            f"set coord_source_shapes in the dataset."
        )

    return OOTGPredictor(
        model=model,
        roles=roles,
        coord_to_latlon=coord_to_latlon,
        grid_shape=native_shape,
        nwp_stride=nwp_stride,
        output_transform=output_transform,
        input_transform=input_transform,
        output_var_names=[str(v) for v in cfg.dataset.obs_vars],
        lr=cfg.get("lr", 5e-4),
        n_samples=cfg.get("n_samples", 20),
        ablate_obs=cfg.get("ablate_obs", False),
        ablate_grid=cfg.get("ablate_grid", False),
    )
