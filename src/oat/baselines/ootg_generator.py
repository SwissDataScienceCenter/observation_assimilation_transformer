import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import xarray as xr

_TNPDIR = Path(__file__).parents[3] / "deps" / "gridded-tnp"
if _TNPDIR.exists() and str(_TNPDIR) not in sys.path:
    sys.path.insert(0, str(_TNPDIR))

from tnp.data.base import DataGenerator, OOTGBatch

from oat.baselines.coord_utils import (
    lonlat_swap,
    lv95_to_latlon,
    nwp_latlon_grid,
    nwp_latlon_grid_from_latlon,
)


class OATOOTGDataGenerator(DataGenerator):
    """Wraps CosmoPeakWeatherDataset or ERA5MeteostatDataset to produce OOTGBatch.

    Accepts the underlying dataset (not TrainDataset/EvalDataset wrappers).
    Each batch is one random time step. Valid stations (all vars observed) are
    split pc_context/1-pc_context into context and target by random permutation.

    Normal variables are standardized; ZeroInflatedLogNormal variables are kept
    in their original non-negative units (standardization would break ZILN support).
    """

    def __init__(
        self,
        dataset,
        obs_mean: torch.Tensor,
        obs_std: torch.Tensor,
        nwp_mean: torch.Tensor,
        nwp_std: torch.Tensor,
        xc_grid_latlon: torch.Tensor,
        coord_to_latlon: Callable[[torch.Tensor], torch.Tensor],
        dist_names: list[str],
        pc_context: float = 0.8,
        nwp_stride: int = 10,
        samples_per_epoch: int = 10000,
        time_indices: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        super().__init__(samples_per_epoch=samples_per_epoch, batch_size=1, **kwargs)
        self.dataset = dataset
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.nwp_mean = nwp_mean
        self.nwp_std = nwp_std
        self.xc_grid_latlon = xc_grid_latlon
        self.coord_to_latlon = coord_to_latlon
        self.dist_names = dist_names
        self.pc_context = pc_context
        self.nwp_stride = nwp_stride
        self.n_nwp = len(dataset.config.nwp_vars)
        self._time_pool = (
            time_indices.tolist()
            if isinstance(time_indices, np.ndarray)
            else (list(time_indices) if time_indices is not None else None)
        )

    def generate_batch(self, batch_shape: torch.Size | None = None) -> OOTGBatch:
        xc, yc, xt, yt, yc_grid = self._sample_once()
        xc_grid = self.xc_grid_latlon.unsqueeze(0)
        return OOTGBatch(
            x=torch.cat([xc, xt], dim=0).unsqueeze(0),
            y=torch.cat([yc, yt], dim=0).unsqueeze(0),
            xc=xc.unsqueeze(0),
            yc=yc.unsqueeze(0),
            xt=xt.unsqueeze(0),
            yt=yt.unsqueeze(0),
            xc_grid=xc_grid,
            yc_grid=yc_grid.unsqueeze(0),
        )

    def _sample_once(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._time_pool is not None:
            idx = self._time_pool[int(torch.randint(len(self._time_pool), ()).item())]
        else:
            idx = int(torch.randint(len(self.dataset), ()).item())

        sample = self.dataset[idx]
        obs = sample["observation_data"]["obs"]        # (N, dim_y)
        mask = sample["observation_data"]["mask_obs"]  # (N, dim_y)
        coords = sample["observation_data"]["coords"]  # (N, 2)
        global_field = sample["global_data"]["global_field"]

        valid = mask.all(dim=-1).bool()
        if valid.sum() < 4:
            return self._sample_once()

        valid_obs = obs[valid].clone()     # (N_valid, dim_y)
        valid_coords = coords[valid]

        # Standardize only Normal-distributed variables
        for v, dn in enumerate(self.dist_names):
            if dn == "normal":
                valid_obs[:, v] = (valid_obs[:, v] - self.obs_mean[v]) / self.obs_std[v]
            # ZILN: keep original non-negative values

        n_valid = int(valid.sum().item())
        n_ctx = max(1, min(int(n_valid * self.pc_context), n_valid - 1))
        perm = torch.randperm(n_valid)
        ctx_idx, tgt_idx = perm[:n_ctx], perm[n_ctx:]

        xc = self.coord_to_latlon(valid_coords[ctx_idx])
        yc = valid_obs[ctx_idx]
        xt = self.coord_to_latlon(valid_coords[tgt_idx])
        yt = valid_obs[tgt_idx]

        nwp_coarse = global_field[: self.n_nwp, :: self.nwp_stride, :: self.nwp_stride]
        nwp_norm = (
            nwp_coarse - self.nwp_mean[:, None, None]
        ) / self.nwp_std[:, None, None]
        yc_grid = nwp_norm.permute(1, 2, 0)

        return xc, yc, xt, yt, yc_grid


def compute_norm_stats(
    dataset,
    n_samples: int = 500,
    time_indices: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate per-variable normalization stats from a random dataset sample.

    Stats are computed for ALL variables. The generator only applies them to
    Normal-distributed variables; ZILN variables ignore these stats at inference.

    Returns:
        obs_mean: (dim_y,) per-variable mean
        obs_std:  (dim_y,) per-variable std, clamped >= 1e-6
        nwp_mean: (n_nwp,) per-channel mean
        nwp_std:  (n_nwp,) per-channel std, clamped >= 1e-6
    """
    n_nwp = len(dataset.config.nwp_vars)
    n_obs = len(dataset.config.obs_vars)

    pool = (
        time_indices.tolist()
        if isinstance(time_indices, np.ndarray)
        else (list(time_indices) if time_indices is not None else list(range(len(dataset))))
    )
    chosen = [pool[i] for i in torch.randperm(len(pool))[: min(n_samples, len(pool))].tolist()]

    obs_vals: list[list[torch.Tensor]] = [[] for _ in range(n_obs)]
    nwp_vals: list[torch.Tensor] = []

    for idx in chosen:
        sample = dataset[int(idx)]
        obs = sample["observation_data"]["obs"]
        mask = sample["observation_data"]["mask_obs"]
        gf = sample["global_data"]["global_field"]
        for v in range(n_obs):
            valid_v = mask[:, v].bool()
            if valid_v.any():
                obs_vals[v].append(obs[valid_v, v])
        nwp_vals.append(gf[:n_nwp].reshape(n_nwp, -1))

    obs_means, obs_stds = [], []
    for v in range(n_obs):
        if obs_vals[v]:
            all_v = torch.cat(obs_vals[v])
            obs_means.append(all_v.mean())
            obs_stds.append(all_v.std().clamp(min=1e-6))
        else:
            obs_means.append(torch.tensor(0.0))
            obs_stds.append(torch.tensor(1.0))

    nwp_all = torch.cat(nwp_vals, dim=1)
    return (
        torch.stack(obs_means),
        torch.stack(obs_stds),
        nwp_all.mean(dim=1),
        nwp_all.std(dim=1).clamp(min=1e-6),
    )


def make_ootg_generator(
    underlying_ds,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    nwp_mean: torch.Tensor,
    nwp_std: torch.Tensor,
    nwp_stride: int | None = None,
    time_indices: np.ndarray | None = None,
    **kwargs,
) -> "OATOOTGDataGenerator":
    """Build OATOOTGDataGenerator for either COSMO or ERA5 dataset.

    Reads dataset_type for coord/grid handling and obs_variable_to_distribution
    for per-variable normalization decisions.
    """
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

    return OATOOTGDataGenerator(
        dataset=underlying_ds,
        obs_mean=obs_mean, obs_std=obs_std,
        nwp_mean=nwp_mean, nwp_std=nwp_std,
        xc_grid_latlon=xc_grid,
        coord_to_latlon=coord_fn,
        dist_names=dist_names,
        nwp_stride=stride,
        time_indices=time_indices,
        **kwargs,
    )
