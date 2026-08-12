from __future__ import annotations

import logging as _logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from anemoi.datasets import open_dataset
from scipy.spatial import cKDTree
from torch.nn.functional import grid_sample
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from oat.data_module import statistics, transform
from oat.data_module.dataset_helpers import (
    ERA5_DERIVED_FEATURES as DERIVED_FEATURES,
    build_obs_variable_support as _build_obs_variable_support,
    dtype_map,
)
from oat.data_module.meteostat_observations import MeteostatObservations

_log = _logging.getLogger(__name__)


@dataclass
class ERA5MeteostatConfig:
    nwp_path: str
    topo_path: str
    obs_path: str
    station_meta_path: str
    nwp_vars: list[str]
    topo_vars: list[str]
    obs_vars: list[str]
    coord_vars: list[str]
    geo_vars: list[str]
    nwp_lat_min: float = 34.0
    nwp_lat_max: float = 72.0
    nwp_lon_min: float = -25.0
    nwp_lon_max: float = 50.0
    nwp_lat_res: float = 0.25
    nwp_lon_res: float = 0.25
    patch_size: tuple[int, int] = (32, 32)
    stations: list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None
    use_time_embedding: bool = True
    dtype: str = "float32"
    catalog: Any = None  # OmegaConf DictConfig with obs_vars bounds
    obs_plot_config: Any = None


class ERA5MeteostatDataset(Dataset[dict[str, torch.Tensor]]):

    def __init__(self, config: Any) -> None:
        self.config = config
        self.catalog = self.config.catalog
        self.obs_var_support = _build_obs_variable_support(
            self.catalog, list(self.config.obs_vars)
        )
        self._load_datasets()
        self._setup_spatial_grids()
        self._load_topo()
        self._load_observations()
        self._mark_shared_memory()

    @property
    def variable_names(self) -> dict[str, dict[str, list[str] | None]]:
        return {
            "global_data": {
                "global_field": list(self.config.nwp_vars)
                + list(self.config.topo_vars),
                "time_emb": (
                    ["sin_day", "cos_day", "sin_year", "cos_year"]
                    if self.config.use_time_embedding
                    else None
                ),
            },
            "observation_data": {
                "coords": list(self.config.coord_vars),
                "geo": list(self.config.geo_vars),
                "topo": list(self.config.topo_vars),
                "obs": list(self.config.obs_vars),
                "mask_obs": list(self.config.obs_vars),
            },
        }

    @property
    def station_names(self) -> list[str]:
        return list(self.stations.tolist())

    @property
    def station_info(self) -> pd.DataFrame:
        return self.obs_ds.stations_table.loc[self.stations]

    def _load_datasets(self) -> None:
        self.nwp_ds = open_dataset(self.config.nwp_path)

        source_vars: list[str] = list(
            dict.fromkeys(
                [v for v in self.config.nwp_vars if v not in DERIVED_FEATURES]
                + [
                    dep
                    for v in self.config.nwp_vars
                    if v in DERIVED_FEATURES
                    for dep in DERIVED_FEATURES[v][0]
                ]
            )
        )
        missing = [v for v in source_vars if v not in self.nwp_ds.name_to_index]
        if missing:
            raise KeyError(
                f"NWP variables (or deps) not in dataset: {missing}. "
                f"Available: {list(self.nwp_ds.name_to_index.keys())}"
            )
        self._source_vars = source_vars
        self.nwp_var_idx = [self.nwp_ds.name_to_index[v] for v in source_vars]

        self.obs_ds = MeteostatObservations(
            obs_path=self.config.obs_path,
            station_meta_path=self.config.station_meta_path,
        )
        self.stations = (
            np.array(list(self.config.stations), dtype=str)
            if self.config.stations is not None
            else self.obs_ds.stations
        )

        all_times = pd.to_datetime(self.nwp_ds.dates).tz_localize("UTC")
        if self.config.start_date is not None or self.config.end_date is not None:
            start = (
                pd.to_datetime(self.config.start_date).tz_localize("UTC")
                if self.config.start_date
                else all_times.min()
            )
            end = (
                pd.to_datetime(self.config.end_date).tz_localize("UTC")
                if self.config.end_date
                else all_times.max()
            )
            mask = (all_times >= start) & (all_times <= end)
            self.times = all_times[mask]
            if len(self.times) == 0:
                raise ValueError(
                    f"No NWP times in [{start}, {end}]. "
                    f"Available: {all_times.min()} - {all_times.max()}"
                )
            self.time_indices = np.where(mask)[0]
        else:
            self.times = all_times
            self.time_indices = np.arange(len(all_times))

        missing_dates = self.nwp_ds.missing
        if missing_dates is not None and len(missing_dates) > 0:
            bad = np.isin(self.time_indices, np.asarray(list(missing_dates)))
            self.times = self.times[~bad]
            self.time_indices = self.time_indices[~bad]

    def _setup_spatial_grids(self) -> None:
        ref_lats = np.arange(
            self.config.nwp_lat_min,
            self.config.nwp_lat_max + 1e-9,
            self.config.nwp_lat_res,
        )
        ref_lons = np.arange(
            self.config.nwp_lon_min,
            self.config.nwp_lon_max + 1e-9,
            self.config.nwp_lon_res,
        )
        self._ref_lats = ref_lats
        self._ref_lons = ref_lons
        self.nwp_shape = (len(ref_lats), len(ref_lons))

        nwp_lons = np.where(
            self.nwp_ds.longitudes > 180,
            self.nwp_ds.longitudes - 360,
            self.nwp_ds.longitudes,
        )
        tree = cKDTree(np.column_stack([nwp_lons, self.nwp_ds.latitudes]))
        grid_lon, grid_lat = np.meshgrid(ref_lons, ref_lats)
        _, self.nwp_indices = tree.query(
            np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
        )

        self.nwp_extent = {
            "min": torch.tensor(
                [float(ref_lons[0]), float(ref_lats[0])], dtype=torch.float32
            ),
            "max": torch.tensor(
                [float(ref_lons[-1]), float(ref_lats[-1])], dtype=torch.float32
            ),
        }

    def _load_topo(self) -> None:
        topo_ds = xr.open_zarr(self.config.topo_path)

        if float(topo_ds.latitude[0]) > float(topo_ds.latitude[-1]):
            topo_ds = topo_ds.isel(latitude=slice(None, None, -1))

        topo_vars = list(self.config.topo_vars)

        self.topo_extent = {
            "min": torch.tensor(
                [float(topo_ds.longitude[0]), float(topo_ds.latitude[0])],
                dtype=torch.float32,
            ),
            "max": torch.tensor(
                [float(topo_ds.longitude[-1]), float(topo_ds.latitude[-1])],
                dtype=torch.float32,
            ),
        }

        _log.info("Loading topo into memory (%d variables) ...", len(topo_vars))
        self.topo_field = torch.from_numpy(
            np.nan_to_num(
                np.stack([topo_ds[v].values for v in topo_vars], axis=0), nan=0.0
            )
        ).to(dtype_map[self.config.dtype])

        nwp_sel = topo_ds.sel(
            latitude=self._ref_lats, longitude=self._ref_lons, method="nearest"
        )
        self.topo_on_nwp_grid = torch.from_numpy(
            np.nan_to_num(
                np.stack([nwp_sel[v].values for v in topo_vars], axis=0), nan=0.0
            )
        ).to(dtype_map[self.config.dtype])

        topo_res = abs(float(topo_ds.latitude[1] - topo_ds.latitude[0]))
        h, w = self.config.patch_size
        gy = torch.linspace(-(h // 2) * topo_res, (h // 2) * topo_res, h)
        gx = torch.linspace(-(w // 2) * topo_res, (w // 2) * topo_res, w)
        lat_off, lon_off = torch.meshgrid(gy, gx, indexing="ij")  # (h, w)
        self.topo_grid = torch.stack([lon_off, lat_off], dim=-1)  # (h, w, 2)

    def _load_observations(self) -> None:
        obs_arr, mask_arr = self.obs_ds.get_observations(
            first_date=self.times[0],
            last_date=self.times[-1] + pd.Timedelta(hours=1),
            parameters=list(self.config.obs_vars),
            stations=self.stations,
            return_mask=True,
            as_numpy=True,
        )
        local_indices = self.time_indices - self.time_indices[0]
        self.obs_tensor = torch.from_numpy(obs_arr[local_indices])
        self.obs_mask_tensor = torch.from_numpy(mask_arr[local_indices])

        obs_vars = list(self.config.obs_vars)
        for wind_var in ("wspd", "wpgt"):
            if wind_var in obs_vars:
                self.obs_tensor[..., obs_vars.index(wind_var)] /= 3.6

        meta = self.obs_ds.stations_table.loc[self.stations]
        self.station_coords = torch.tensor(
            meta[["longitude", "latitude"]].values.copy(),
            dtype=dtype_map[self.config.dtype],
        )
        self.station_geo = self._get_geo(self.station_coords)
        self.obs_ds._obs = None  # free 17.6 GB DataFrame; obs_tensor is in shared memory

    def _mark_shared_memory(self) -> None:
        for attr in (
            "topo_field", "topo_on_nwp_grid", "station_coords", "station_geo",
            "obs_tensor", "obs_mask_tensor",
        ):
            if hasattr(self, attr):
                getattr(self, attr).share_memory_()

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        time = self.times[idx]
        nwp_idx = self.time_indices[idx]

        raw = self.nwp_ds[int(nwp_idx)][self.nwp_var_idx]
        raw = raw.reshape(len(self.nwp_var_idx), -1)
        raw_grid = torch.from_numpy(
            raw[:, self.nwp_indices].reshape(len(self.nwp_var_idx), *self.nwp_shape)
        ).to(dtype_map[self.config.dtype])

        src = {var: raw_grid[i] for i, var in enumerate(self._source_vars)}
        nwp_channels = [
            (
                DERIVED_FEATURES[v][1](*[src[d] for d in DERIVED_FEATURES[v][0]])
                if v in DERIVED_FEATURES
                else src[v]
            )
            for v in self.config.nwp_vars
        ]
        global_field = torch.cat(
            [torch.stack(nwp_channels, dim=0), self.topo_on_nwp_grid], dim=0
        )

        global_data: dict[str, torch.Tensor] = {"global_field": global_field}
        if self.config.use_time_embedding:
            global_data["time_emb"] = self._get_time_embedding(time)

        observation_data = {
            "coords": self.station_coords,
            "geo": self.station_geo,
            "topo": self._extract_patches(
                self.station_coords, self.topo_field, self.topo_grid, self.topo_extent
            ),
            "obs": self.obs_tensor[idx].to(dtype_map[self.config.dtype]).nan_to_num(nan=0.0),
            "mask_obs": self.obs_mask_tensor[idx].to(dtype_map[self.config.dtype]),
        }

        return {
            "index": idx,
            "global_data": global_data,
            "observation_data": observation_data,
        }

    def get_query(self, query_coords: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "coords": query_coords,
            "topo": self._extract_patches(
                query_coords, self.topo_field, self.topo_grid, self.topo_extent
            ),
            "geo": self._get_geo(query_coords),
        }

    def _extract_patches(
        self,
        positions: torch.Tensor,
        field: torch.Tensor,
        grid: torch.Tensor,
        extent: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        grid_pos = positions.unsqueeze(1).unsqueeze(1) + grid
        return grid_sample(
            field.unsqueeze(0).expand(positions.shape[0], -1, -1, -1),
            self._normalize_positions(grid_pos, extent["min"], extent["max"]),
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )

    def _get_geo(self, coords: torch.Tensor) -> torch.Tensor:
        zero_grid = torch.zeros(1, 1, 2, dtype=coords.dtype, device=coords.device)
        elev = self._extract_patches(
            coords, self.topo_field[0:1], zero_grid, self.topo_extent
        )[:, 0, 0, 0]
        return torch.cat([coords, elev.unsqueeze(1)], dim=1)

    @staticmethod
    def _normalize_positions(
        pos: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> torch.Tensor:
        return 2.0 * (pos - min_val) / (max_val - min_val) - 1.0

    def _get_time_embedding(self, dt: pd.Timestamp) -> torch.Tensor:
        dt = pd.to_datetime(dt)
        secs = dt.hour * 3600 + dt.minute * 60 + dt.second
        doy = dt.dayofyear + secs / 86400.0
        return torch.tensor(
            [
                np.sin(2 * np.pi * secs / 86400.0),
                np.cos(2 * np.pi * secs / 86400.0),
                np.sin(2 * np.pi * doy / 365.2425),
                np.cos(2 * np.pi * doy / 365.2425),
            ],
            dtype=dtype_map[self.config.dtype],
        )

    def get_transform(
        self,
        num_data_points: int,
        dataloader: DataLoader[dict[str, torch.Tensor]] | None = None,
        type: str = "real",
        data_indices: np.ndarray[Any, np.dtype[np.intp]] | None = None,
    ) -> tuple[transform.NestedTransform, transform.Transform]:

        def _collect_stats(
            pre_t: transform.NestedTransform,
            dl: DataLoader[dict[str, torch.Tensor]] | None,
            n: int | None,
        ) -> dict[str, Any]:
            num_points = n if n is not None else len(self)
            stats: dict[str, Any] = {
                "observation_data": {
                    "obs": statistics.RunningFeatureStats(
                        len(list(self.config.obs_vars)), feature_dim=-1
                    ),
                    "topo": statistics.RunningFeatureStats(
                        len(list(self.config.topo_vars)), feature_dim=-3
                    ),
                    "geo": statistics.RunningFeatureStats(
                        self.station_geo.shape[-1], feature_dim=-1
                    ),
                },
                "global_data": {
                    "global_field": statistics.RunningFeatureStats(
                        len(list(self.config.nwp_vars))
                        + len(list(self.config.topo_vars)),
                        feature_dim=-3,
                    ),
                },
            }

            def _gather(
                sample: dict[str, Any] | torch.Tensor,
                gatherer: statistics.RunningFeatureStats | dict[str, Any],
            ) -> None:
                if isinstance(gatherer, statistics.RunningFeatureStats):
                    gatherer.update(sample)  # type: ignore[arg-type]
                elif isinstance(gatherer, dict):
                    for k, v in gatherer.items():
                        _gather(sample[k], v)  # type: ignore[index]

            if dl is not None:
                bs = dl.batch_size
                for i, batch in tqdm(
                    enumerate(dl), total=num_points // bs, desc="Gathering statistics"  # type: ignore[operator]
                ):
                    if i * bs >= num_points:
                        break
                    _gather(pre_t(batch), stats)
            else:
                pool = (
                    data_indices if data_indices is not None else np.arange(len(self))
                )
                for idx in tqdm(
                    pool[torch.randperm(len(pool))[:num_points].numpy()],
                    desc="Gathering statistics",
                ):
                    _gather(pre_t(self[int(idx)]), stats)
            return stats

        offset = 1e-3

        if type == "real":
            log_indices = [
                i
                for i, v in enumerate(self.config.obs_vars)
                if self.obs_var_support.lower_bounds[i] >= 0.0
                and self.obs_var_support.upper_bounds[i] == float("inf")
            ]
            log_t = transform.LogTransform(
                apply_to=log_indices, feature_dim=-1, offset=offset
            )
            pre_t = transform.NestedTransform({"observation_data": {"obs": log_t}})
            stats = _collect_stats(pre_t, dataloader, num_data_points)

            topo_t = stats["observation_data"]["topo"].get_standard_transform()
            geo_t = stats["observation_data"]["geo"].get_standard_transform()
            gf_t = stats["global_data"]["global_field"].get_standard_transform()
            output_transform = transform.ComposedTransform(
                [log_t, stats["observation_data"]["obs"].get_standard_transform()]
            )
            post_transform = transform.NestedTransform(
                {
                    "observation_data": {
                        "obs": output_transform,
                        "topo": topo_t,
                        "geo": geo_t,
                    },
                    "query_data": {
                        "obs": output_transform,
                        "topo": topo_t,
                        "geo": geo_t,
                    },
                    "global_data": {
                        "global_field": gf_t,
                    },
                }
            )
        elif type == "keep_support":
            # Shift strictly-positive variables away from zero before standardising
            # so that the support constraint is preserved after the inverse transform.
            apply_offset_vars = ["wspd", "wpgt"]
            offset_indices = [
                i for i, v in enumerate(self.config.obs_vars) if v in apply_offset_vars
            ]
            offset_arr = torch.zeros(len(list(self.config.obs_vars)))
            offset_arr[offset_indices] = offset
            scale_arr = torch.ones(len(list(self.config.obs_vars)))
            offset_t = transform.TranslateScaleTransform(
                translate=offset_arr, scale=scale_arr, channel_dim=-1
            )
            pre_t = transform.NestedTransform({"observation_data": {"obs": offset_t}})
            stats = _collect_stats(pre_t, dataloader, num_data_points)

            translate_arr = -stats["observation_data"]["obs"].mean.clone()
            # Variables with zero lower bound and infinite upper bound are only
            # scaled (not translated) to preserve their non-negativity.
            only_scale_indices = [
                i
                for i in range(len(list(self.config.obs_vars)))
                if self.obs_var_support.lower_bounds[i] == 0.0
                and self.obs_var_support.upper_bounds[i] == float("inf")
            ]
            translate_arr[only_scale_indices] = 0.0
            scale_arr = 1.0 / stats["observation_data"]["obs"].std.clone()

            output_transform = transform.ComposedTransform(
                [
                    offset_t,
                    transform.TranslateScaleTransform(
                        translate=translate_arr, scale=scale_arr, channel_dim=-1
                    ),
                ]
            )
            topo_t = stats["observation_data"]["topo"].get_standard_transform()
            geo_t = stats["observation_data"]["geo"].get_standard_transform()
            gf_t = stats["global_data"]["global_field"].get_standard_transform()
            post_transform = transform.NestedTransform(
                {
                    "observation_data": {
                        "obs": output_transform,
                        "topo": topo_t,
                        "geo": geo_t,
                    },
                    "query_data": {
                        "obs": output_transform,
                        "topo": topo_t,
                        "geo": geo_t,
                    },
                    "global_data": {
                        "global_field": gf_t,
                    },
                }
            )
        else:
            raise ValueError(
                f"Unsupported transform type: {type!r}. "
                "Supported: 'real', 'keep_support'."
            )

        return post_transform, output_transform

    def get_baseline_pre_transform(
        self,
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        nwp_vars_set = set(self.config.nwp_vars)
        if "wind_speed" in nwp_vars_set:
            _wind_deps: list[str] = ["wind_speed"]
            _wind_fn: Callable[[dict[str, Any]], torch.Tensor] = lambda nwp: nwp["wind_speed"]
        else:
            _wind_deps = ["10u", "10v"]
            _wind_fn = lambda nwp: torch.sqrt(nwp["10u"] ** 2 + nwp["10v"] ** 2)

        _mapping: dict[
            str, tuple[list[str], Callable[[dict[str, Any]], torch.Tensor]]
        ] = {
            "temp": (["2t"], lambda nwp: nwp["2t"] - 273.15),
            "wspd": (_wind_deps, _wind_fn),
            "wpgt": (_wind_deps, _wind_fn),
            "prcp": (["tp"], lambda nwp: nwp["tp"] * 1000.0),
        }
        for obs_var in self.config.obs_vars:
            if obs_var not in _mapping:
                raise ValueError(f"No ERA5 mapping for obs variable {obs_var!r}.")
            deps, _ = _mapping[obs_var]
            missing = [d for d in deps if d not in self.config.nwp_vars]
            if missing:
                raise ValueError(
                    f"ERA5 deps {missing} for {obs_var!r} not in config.nwp_vars."
                )

        nwp_vars = list(self.config.nwp_vars)
        obs_vars = list(self.config.obs_vars)

        def _pre(sample: dict[str, Any]) -> dict[str, Any]:
            nwp_fields = sample["global_data"]["global_field"]
            nwp = {v: nwp_fields[..., i, :, :] for i, v in enumerate(nwp_vars)}
            new_field = torch.stack([_mapping[v][1](nwp) for v in obs_vars], dim=-3)

            return {
                **sample,
                "global_data": {**sample["global_data"], "global_field": new_field},
            }

        return _pre
