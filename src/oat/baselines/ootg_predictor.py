import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import torch

_TNPDIR = Path(__file__).parents[3] / "deps" / "gridded-tnp"
if _TNPDIR.exists() and str(_TNPDIR) not in sys.path:
    sys.path.insert(0, str(_TNPDIR))

from tnp.models.gridded_tnp import OOTGGriddedTNP

from oat.baselines.coord_utils import (
    lonlat_swap,
    lv95_to_latlon,
    nwp_latlon_grid,
    nwp_latlon_grid_from_latlon,
)
from oat.baselines.ootg_likelihood import JointMixedDistribution
from oat.data_module.transform import Transform
from oat.predictor_module.base_predictor import BasePredictor
from oat.predictor_module.functional_metrics import mc_crps, point_mae


class OOTGPredictor(BasePredictor):
    """Wraps a trained OOTGGriddedTNP to evaluate on OAT-format batches.

    At inference, ALL valid observation stations are used as context.
    Predictions are returned in original units:
      - Normal variables: model predicts in standardized space, output is denormalized
      - ZeroInflatedLogNormal variables: model predicts in original non-negative space

    Outputs [B, N_q, n_samples, n_vars], matching BasePredictor's predict_step contract.
    For n_samples > 1, uses dist.sample() (ZILN gate is Bernoulli → not reparameterizable).
    """

    def __init__(
        self,
        model: OOTGGriddedTNP,
        obs_vars: list[str],
        obs_mean: torch.Tensor,  # (dim_y,)
        obs_std: torch.Tensor,  # (dim_y,)
        nwp_mean: torch.Tensor,  # (n_nwp,)
        nwp_std: torch.Tensor,  # (n_nwp,)
        xc_grid_base: torch.Tensor,  # (H_c, W_c, 2)
        coord_to_latlon: Callable[[torch.Tensor], torch.Tensor],
        dist_names: list[str],
        n_nwp: int,
        nwp_stride: int,
        output_transform: "Transform | None" = None,
        lr: float = 5e-4,
        n_samples: int = 50,
        use_topo: bool = False,
        topo_transform: "Transform | None" = None,
    ) -> None:
        super().__init__(output_var_names=obs_vars)
        self.model = model
        self.register_buffer("obs_mean", obs_mean)
        self.register_buffer("obs_std", obs_std)
        self.register_buffer("nwp_mean", nwp_mean)
        self.register_buffer("nwp_std", nwp_std)
        self.register_buffer("xc_grid_base", xc_grid_base)
        self.coord_to_latlon = coord_to_latlon
        self.dist_names = dist_names
        self.n_nwp = n_nwp
        self.nwp_stride = nwp_stride
        self.metrics_fn = {"point_mae": point_mae, "crps": mc_crps}
        self.output_transform = output_transform
        self.lr = lr
        self.n_samples = n_samples
        self.use_topo = use_topo
        self.topo_transform = topo_transform

    def _normalize_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return batch

    def _build_yc(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply per-variable normalization. Normal → standardize; ZILN → identity."""
        yc_obs = obs.clone()
        if self.output_transform is not None:
            yc_obs = self.output_transform(yc_obs)
        else:
            # legacy path: manual loop using obs_mean / obs_std
            om = self.obs_mean.to(obs.device)
            os_ = self.obs_std.to(obs.device)
            for v, dn in enumerate(self.dist_names):
                if dn == "normal":
                    yc_obs[:, v] = (yc_obs[:, v] - om[v]) / os_[v]
        return yc_obs

    def _denormalize(self, mean: torch.Tensor) -> torch.Tensor:
        """Denormalize predictions: Normal vars → original units; ZILN → unchanged."""
        if self.output_transform is not None:
            return self.output_transform.inverse(mean)
        # legacy path
        out = mean.clone()
        om = self.obs_mean.to(mean.device)
        os_ = self.obs_std.to(mean.device)
        for v, dn in enumerate(self.dist_names):
            if dn == "normal":
                out[..., v] = mean[..., v] * os_[v] + om[v]
        return out

    def _dist_for_sample(
        self,
        obs: torch.Tensor,  # (N_obs, dim_y)
        mask: torch.Tensor,  # (N_obs, dim_y)
        coords: torch.Tensor,  # (N_obs, 2) dataset CRS
        q_coords: torch.Tensor,  # (N_q, 2) dataset CRS
        global_field: torch.Tensor,  # (C_global, H, W)
        topo_obs: torch.Tensor | None = None,  # (N_obs, C_topo, Hp, Wp)
        topo_q: torch.Tensor | None = None,  # (N_q, C_topo, Hp, Wp)
    ) -> JointMixedDistribution:
        valid = mask.all(dim=-1).bool()
        dev = global_field.device

        if valid.sum() == 0:
            # No valid context: pass empty context tensors
            xc = torch.zeros(1, 0, 2, device=dev, dtype=global_field.dtype)
            yc = torch.zeros(
                1, 0, len(self.output_var_names), device=dev, dtype=global_field.dtype
            )
        else:
            xc = self.coord_to_latlon(coords[valid]).unsqueeze(0).to(dev)
            yc = self._build_yc(obs[valid]).unsqueeze(0).to(dev)

        xt = self.coord_to_latlon(q_coords).unsqueeze(0).to(dev)

        nwp_c = global_field[: self.n_nwp, :: self.nwp_stride, :: self.nwp_stride]
        nm = self.nwp_mean.to(dev)
        ns = self.nwp_std.to(dev)
        yc_grid = (
            ((nwp_c - nm[:, None, None]) / ns[:, None, None])
            .permute(1, 2, 0)
            .unsqueeze(0)
        )
        xc_grid = self.xc_grid_base.to(dev).unsqueeze(0)

        if self.use_topo:
            assert self.topo_transform is not None and topo_q is not None
            tt = self.topo_transform
            if valid.sum() == 0:
                topo_c = torch.zeros(
                    1, 0, *topo_q.shape[-3:], device=dev, dtype=global_field.dtype
                )
            else:
                assert topo_obs is not None
                topo_c = tt(topo_obs[valid]).unsqueeze(0).to(dev)
            topo_t = tt(topo_q).unsqueeze(0).to(dev)
            return self.model(xc, yc, xc_grid, yc_grid, xt, topo_c, topo_t)
        return self.model(xc, yc, xc_grid, yc_grid, xt)

    def _denorm_samples(self, dist: "JointMixedDistribution", n: int) -> torch.Tensor:
        """Draw n samples from a single-batch dist → (N_q, n, dim_y) in physical units."""
        raw = dist.sample(torch.Size([n]))  # (n, 1, N_q, dim_y)
        raw = raw.squeeze(-3) if raw.ndim == 4 else raw  # → (n, N_q, dim_y)
        raw = raw.permute(1, 0, 2)  # → (N_q, n, dim_y)
        return self._denormalize(raw)

    def forward(self, batch: dict[str, Any]):
        obs_b = batch["observation_data"]["obs"]
        mask_b = batch["observation_data"]["mask_obs"]
        coords_b = batch["observation_data"]["coords"]
        q_coords_b = batch["query_data"]["coords"]
        gf_b = batch["global_data"]["global_field"]
        topo_obs_b = (
            batch["observation_data"].get("topo", None) if self.use_topo else None
        )
        topo_q_b = batch["query_data"].get("topo", None) if self.use_topo else None
        target_obs = batch["query_data"].get("obs", None)
        target_mask = batch["query_data"].get("mask_obs", None)

        if obs_b.ndim == 2:
            obs_b = obs_b.unsqueeze(0)
            mask_b = mask_b.unsqueeze(0)
            coords_b = coords_b.unsqueeze(0)
            q_coords_b = q_coords_b.unsqueeze(0)
            gf_b = gf_b.unsqueeze(0)
            if topo_obs_b is not None and topo_q_b is not None:
                topo_obs_b = topo_obs_b.unsqueeze(0)
                topo_q_b = topo_q_b.unsqueeze(0)
            if target_obs is not None:
                target_obs = target_obs.unsqueeze(0)
            if target_mask is not None:
                target_mask = target_mask.unsqueeze(0)

        # Metrics (CRPS/MAE) are computed on drawn samples, matching
        # OATPredictor: never on a single mean sample (that degenerates mc_crps
        # to a 0/0 = NaN). self.n_samples is the OOTG analog of model.num_samples.
        preds = []
        for b in range(obs_b.shape[0]):
            dist = self._dist_for_sample(
                obs_b[b],
                mask_b[b],
                coords_b[b],
                q_coords_b[b],
                gf_b[b],
                topo_obs=None if topo_obs_b is None else topo_obs_b[b],
                topo_q=None if topo_q_b is None else topo_q_b[b],
            )
            preds.append(self._denorm_samples(dist, self.n_samples))

        return torch.stack(preds, dim=0), target_obs, target_mask  # (B, N_q, n, dim_y)

    def _forward_generator_batch(self, batch) -> "JointMixedDistribution":
        if self.use_topo:
            return self.model(
                batch.xc,
                batch.yc,
                batch.xc_grid,
                batch.yc_grid,
                batch.xt,
                batch.topo_c,
                batch.topo_t,
            )
        return self.model(batch.xc, batch.yc, batch.xc_grid, batch.yc_grid, batch.xt)

    def _draw_samples(self, dist: "JointMixedDistribution") -> torch.Tensor:
        """Return [B, N, n_samples, dim_y] in normalized space."""
        raw = dist.sample(torch.Size([self.n_samples]))  # [n_samples, B, N, dim_y]
        return raw.permute(1, 2, 0, 3)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        dist = self._forward_generator_batch(batch)
        mask = torch.ones_like(batch.yt)

        # Normalized analytical CRPS — differentiable optimizer signal
        crps_norm = dist.crps(batch.yt)  # [B, N, dim_y]
        per_var_crps = (crps_norm * mask).mean(dim=(0, 1))  # [dim_y]
        for i, name in enumerate(self.output_var_names):
            self.log(f"train_loss_{name}", per_var_crps[i], on_step=True, on_epoch=True)
        loss = per_var_crps.mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Physical-space metrics — no gradient needed
        with torch.no_grad():
            samples_norm = self._draw_samples(dist)
            samples_phys = self._denormalize(samples_norm)
            yt_phys = self._denormalize(batch.yt)
            self._log_instant_metrics(
                samples_phys,
                yt_phys,
                mask,
                prefix="train",
                on_step=True,
                on_epoch=True,
            )
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        if isinstance(batch, dict):
            # Inference path: dict-style batch from standard evaluation dataset
            y_pred, target_obs, target_mask = self(batch)  # (B, N_q, n, dim_y)
            self._log_instant_metrics(
                y_pred,
                target_obs,
                target_mask,
                prefix="val",
                on_step=False,
                on_epoch=True,
            )
        else:
            # Training path: generator batch (dot-notation attributes)
            dist = self._forward_generator_batch(batch)
            mask = torch.ones_like(batch.yt)

            # Normalized analytical CRPS — consistent with training_step
            crps_norm = dist.crps(batch.yt)  # [B, N, dim_y]
            per_var_crps = (crps_norm * mask).mean(dim=(0, 1))  # [dim_y]
            for i, name in enumerate(self.output_var_names):
                self.log(
                    f"val_loss_{name}", per_var_crps[i], on_step=False, on_epoch=True
                )
            self.log(
                "val_loss",
                per_var_crps.mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            with torch.no_grad():
                samples_norm = self._draw_samples(dist)
                samples_phys = self._denormalize(samples_norm)
                yt_phys = self._denormalize(batch.yt)
                self._log_instant_metrics(
                    samples_phys,
                    yt_phys,
                    mask,
                    prefix="val",
                    on_step=False,
                    on_epoch=True,
                )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        y_pred, target_obs, target_mask = self(batch)  # (B, N_q, n, dim_y)
        self._log_instant_metrics(y_pred, target_obs, target_mask, prefix="test")

    def predict_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
        return_as: Literal["samples", "distribution"] = "samples",
        n_samples: int | None = None,
    ) -> torch.Tensor:
        if return_as != "samples":
            raise ValueError("OOTGPredictor only supports return_as='samples'.")
        n = n_samples if n_samples is not None else 1

        obs_b = batch["observation_data"]["obs"]
        mask_b = batch["observation_data"]["mask_obs"]
        coords_b = batch["observation_data"]["coords"]
        q_coords_b = batch["query_data"]["coords"]
        gf_b = batch["global_data"]["global_field"]
        topo_obs_b = (
            batch["observation_data"].get("topo", None) if self.use_topo else None
        )
        topo_q_b = batch["query_data"].get("topo", None) if self.use_topo else None

        if obs_b.ndim == 2:
            obs_b = obs_b.unsqueeze(0)
            mask_b = mask_b.unsqueeze(0)
            coords_b = coords_b.unsqueeze(0)
            q_coords_b = q_coords_b.unsqueeze(0)
            gf_b = gf_b.unsqueeze(0)
            if topo_obs_b is not None and topo_q_b is not None:
                topo_obs_b = topo_obs_b.unsqueeze(0)
                topo_q_b = topo_q_b.unsqueeze(0)

        with torch.no_grad():
            all_samples = []
            for b in range(obs_b.shape[0]):
                dist = self._dist_for_sample(
                    obs_b[b],
                    mask_b[b],
                    coords_b[b],
                    q_coords_b[b],
                    gf_b[b],
                    topo_obs=None if topo_obs_b is None else topo_obs_b[b],
                    topo_q=None if topo_q_b is None else topo_q_b[b],
                )
                if n == 1:
                    # (1, Nq, dim_y) → squeeze batch → (Nq, dim_y) → unsqueeze sample dim
                    s = self._denormalize(dist.mean.squeeze(0)).unsqueeze(
                        -2
                    )  # (Nq, 1, dim_y)
                else:
                    s = self._denorm_samples(dist, n)  # (Nq, n, dim_y)
                all_samples.append(s)
        return torch.stack(all_samples, dim=0)  # (B, Nq, n, dim_y)

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


def create_ootg_predictor(
    model: OOTGGriddedTNP,
    underlying_ds,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    nwp_mean: torch.Tensor,
    nwp_std: torch.Tensor,
    nwp_stride: int | None = None,
    output_transform: "Transform | None" = None,
) -> OOTGPredictor:
    """Build OOTGPredictor from a trained model and the underlying dataset."""
    import xarray as xr

    dt = underlying_ds.config.dataset_type
    if dt == "cosmo_peakweather":
        stride = nwp_stride if nwp_stride is not None else 10
        ref = xr.open_zarr(underlying_ds.config.ref_nwp_path)
        xc_grid = nwp_latlon_grid(ref.x.values, ref.y.values, stride)
        coord_fn = lv95_to_latlon
    elif dt == "era5_meteostat":
        stride = nwp_stride if nwp_stride is not None else 4
        xc_grid = nwp_latlon_grid_from_latlon(
            underlying_ds._ref_lats, underlying_ds._ref_lons, stride
        )
        coord_fn = lonlat_swap
    else:
        raise ValueError(f"Unknown dataset_type: {dt!r}")

    v2d = dict(underlying_ds.config.obs_variable_to_distribution)
    dist_names = [v2d.get(v, "normal") for v in underlying_ds.config.obs_vars]

    return OOTGPredictor(
        model=model,
        obs_vars=list(underlying_ds.config.obs_vars),
        obs_mean=obs_mean,
        obs_std=obs_std,
        nwp_mean=nwp_mean,
        nwp_std=nwp_std,
        xc_grid_base=xc_grid,
        coord_to_latlon=coord_fn,
        dist_names=dist_names,
        n_nwp=len(underlying_ds.config.nwp_vars),
        nwp_stride=stride,
        output_transform=output_transform,
    )
