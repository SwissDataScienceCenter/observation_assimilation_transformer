import sys
from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass
class OOTGTopoBatch(OOTGBatch):
    """OOTGBatch carrying high-res topo patches for context and target tokens.

    topo_c: [B, Nc, C_topo, Hp, Wp]
    topo_t: [B, Nt, C_topo, Hp, Wp]
    """

    topo_c: torch.Tensor
    topo_t: torch.Tensor


class OATOOTGDataGenerator(DataGenerator):
    """Wraps CosmoPeakWeatherDataset or ERA5MeteostatDataset to produce OOTGBatch.

    Accepts the underlying dataset (not TrainDataset/EvalDataset wrappers).
    Each batch is one random time step. Station eligibility for context/target
    is controlled by either `eligible_station_indices` (a pool from which
    context and target are drawn by random pc_context/1-pc_context permutation
    each sample, mirroring TrainDataset) or `context_station_indices` +
    `target_station_indices` (fixed disjoint roles, mirroring EvalDataset). If
    none of these are given, all stations in the sample are eligible and
    randomly split (legacy behavior, station-split-agnostic).

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
        output_transform=None,
        eligible_station_indices: np.ndarray | None = None,
        context_station_indices: np.ndarray | None = None,
        target_station_indices: np.ndarray | None = None,
        ablate_obs: bool = False,
        ablate_grid: bool = False,
        topo_transform=None,
        **kwargs,
    ) -> None:
        super().__init__(samples_per_epoch=samples_per_epoch, batch_size=1, **kwargs)
        if (context_station_indices is None) != (target_station_indices is None):
            raise ValueError(
                "context_station_indices and target_station_indices must be "
                "given together."
            )
        if context_station_indices is not None and eligible_station_indices is not None:
            raise ValueError(
                "Pass either eligible_station_indices (random split) or "
                "context_station_indices/target_station_indices (fixed roles), "
                "not both."
            )
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
        self.output_transform = output_transform
        # Ablations for H2 (NWP-grid) vs H3 (observation-assimilation) diagnosis.
        # ablate_obs: drop all station context -> grid built from NWP cells only.
        # ablate_grid: zero the NWP grid values -> pseudo-token grid carries only
        # assimilated station observations (positional grid retained).
        self.ablate_obs = ablate_obs
        self.ablate_grid = ablate_grid
        self.topo_transform = topo_transform
        self.n_nwp = len(dataset.config.nwp_vars) + len(dataset.config.topo_vars)
        self.eligible_station_indices = (
            torch.as_tensor(eligible_station_indices, dtype=torch.long)
            if eligible_station_indices is not None
            else None
        )
        self.context_station_indices = (
            torch.as_tensor(context_station_indices, dtype=torch.long)
            if context_station_indices is not None
            else None
        )
        self.target_station_indices = (
            torch.as_tensor(target_station_indices, dtype=torch.long)
            if target_station_indices is not None
            else None
        )
        self._time_pool = (
            time_indices.tolist()
            if isinstance(time_indices, np.ndarray)
            else (list(time_indices) if time_indices is not None else None)
        )

    def generate_batch(self, batch_shape: torch.Size | None = None) -> "OOTGBatch":
        xc, yc, xt, yt, yc_grid, topo_c, topo_t = self._sample_once()
        xc_grid = self.xc_grid_latlon.unsqueeze(0)
        common = dict(
            x=torch.cat([xc, xt], dim=0).unsqueeze(0),
            y=torch.cat([yc, yt], dim=0).unsqueeze(0),
            xc=xc.unsqueeze(0),
            yc=yc.unsqueeze(0),
            xt=xt.unsqueeze(0),
            yt=yt.unsqueeze(0),
            xc_grid=xc_grid,
            yc_grid=yc_grid.unsqueeze(0),
        )
        if self.topo_transform is not None:
            return OOTGTopoBatch(
                **common, topo_c=topo_c.unsqueeze(0), topo_t=topo_t.unsqueeze(0)
            )
        return OOTGBatch(**common)

    def _sample_once(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self._time_pool is not None:
            idx = self._time_pool[int(torch.randint(len(self._time_pool), ()).item())]
        else:
            idx = int(torch.randint(len(self.dataset), ()).item())

        sample = self.dataset[idx]
        obs = sample["observation_data"]["obs"]  # (N, dim_y)
        mask = sample["observation_data"]["mask_obs"]  # (N, dim_y)
        coords = sample["observation_data"]["coords"]  # (N, 2)
        global_field = sample["global_data"]["global_field"]

        valid = mask.all(dim=-1).bool()
        n_stations = valid.shape[0]

        fixed_roles = self.context_station_indices is not None
        if fixed_roles:
            ctx_eligible = torch.zeros(n_stations, dtype=torch.bool)
            ctx_eligible[self.context_station_indices] = True
            tgt_eligible = torch.zeros(n_stations, dtype=torch.bool)
            tgt_eligible[self.target_station_indices] = True
            combined_valid = valid & (ctx_eligible | tgt_eligible)
        else:
            combined_valid = valid
            if self.eligible_station_indices is not None:
                eligible = torch.zeros(n_stations, dtype=torch.bool)
                eligible[self.eligible_station_indices] = True
                combined_valid = combined_valid & eligible
            if combined_valid.sum() < 4:
                return self._sample_once()

        valid_obs = obs[combined_valid].clone()  # (N_valid, dim_y)
        valid_coords = coords[combined_valid]

        topo_valid = None
        if self.topo_transform is not None:
            topo_all = sample["observation_data"]["topo"]  # (N, C_topo, Hp, Wp)
            # topo_transform may share buffers with OOTGPredictor, which Lightning
            # moves to GPU; bring sel to the transform's device then back to CPU.
            sel = topo_all[combined_valid]
            buf = next(self.topo_transform.buffers(), None)
            src_dev = sel.device
            topo_valid = self.topo_transform(
                sel.to(buf.device) if buf is not None else sel
            ).to(src_dev)

        # Standardize only Normal-distributed variables
        if self.output_transform is not None:
            # output_transform may share buffers with OOTGPredictor, which Lightning
            # moves to GPU; bring valid_obs to the transform's device then back to CPU.
            buf = next(self.output_transform.buffers(), None)
            src_dev = valid_obs.device
            valid_obs = self.output_transform(
                valid_obs.to(buf.device) if buf is not None else valid_obs
            ).to(src_dev)
        else:
            for v, dn in enumerate(self.dist_names):
                if dn == "normal":
                    valid_obs[:, v] = (
                        valid_obs[:, v] - self.obs_mean[v]
                    ) / self.obs_std[v]
                # ZILN: keep original non-negative values

        if fixed_roles:
            valid_positions = combined_valid.nonzero(as_tuple=True)[0]
            ctx_idx = ctx_eligible[valid_positions].nonzero(as_tuple=True)[0]
            tgt_idx = tgt_eligible[valid_positions].nonzero(as_tuple=True)[0]
            if len(ctx_idx) < 1 or len(tgt_idx) < 1:
                return self._sample_once()
        else:
            n_valid = valid_obs.shape[0]
            n_ctx = max(1, min(int(n_valid * self.pc_context), n_valid - 1))
            perm = torch.randperm(n_valid)
            ctx_idx, tgt_idx = perm[:n_ctx], perm[n_ctx:]

        xc = self.coord_to_latlon(valid_coords[ctx_idx])
        yc = valid_obs[ctx_idx]
        xt = self.coord_to_latlon(valid_coords[tgt_idx])
        yt = valid_obs[tgt_idx]

        nwp_coarse = global_field[: self.n_nwp, :: self.nwp_stride, :: self.nwp_stride]
        nwp_norm = (nwp_coarse - self.nwp_mean[:, None, None]) / self.nwp_std[
            :, None, None
        ]
        yc_grid = nwp_norm.permute(1, 2, 0)

        # Ablations: applied identically at train and eval so the model is both
        # trained and scored under the same information regime.
        if self.ablate_obs:
            # Drop station context; targets (xt, yt) unchanged. Grid is built
            # from NWP cells only. Keeps xc/yc 2D with zero rows.
            xc = xc[:0]
            yc = yc[:0]
        if self.ablate_grid:
            yc_grid = torch.zeros_like(yc_grid)

        if topo_valid is not None:
            topo_c = topo_valid[ctx_idx]
            topo_t = topo_valid[tgt_idx]
            if self.ablate_obs:
                topo_c = topo_c[:0]
        else:
            # No topo requested: return empty placeholders (never read downstream).
            topo_c = torch.empty(0)
            topo_t = torch.empty(0)

        return xc, yc, xt, yt, yc_grid, topo_c, topo_t


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
    n_nwp = len(dataset.config.nwp_vars) + len(dataset.config.topo_vars)
    n_obs = len(dataset.config.obs_vars)

    pool = (
        time_indices.tolist()
        if isinstance(time_indices, np.ndarray)
        else (
            list(time_indices)
            if time_indices is not None
            else list(range(len(dataset)))
        )
    )
    chosen = [
        pool[i] for i in torch.randperm(len(pool))[: min(n_samples, len(pool))].tolist()
    ]

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
    output_transform=None,
    ablate_obs: bool = False,
    ablate_grid: bool = False,
    topo_transform=None,
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
        obs_mean=obs_mean,
        obs_std=obs_std,
        nwp_mean=nwp_mean,
        nwp_std=nwp_std,
        xc_grid_latlon=xc_grid,
        coord_to_latlon=coord_fn,
        dist_names=dist_names,
        nwp_stride=stride,
        time_indices=time_indices,
        output_transform=output_transform,
        ablate_obs=ablate_obs,
        ablate_grid=ablate_grid,
        topo_transform=topo_transform,
        **kwargs,
    )
