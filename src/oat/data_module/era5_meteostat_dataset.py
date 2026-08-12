"""ERA5-Meteostat dataset that emits a dict of ``Field`` per sample.

Thin concrete subclass of ``DatasetBase``: reuses the existing ERA5 NWP /
topography / Meteostat-observation loaders from
``legacy/era5_meteostat_dataset.py`` and wires them into three ``Source``
objects (obs, nwp, topo) consumed by the config-driven field assembly in
``DatasetBase``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import xarray as xr
from omegaconf import OmegaConf
from torch import Tensor

from oat.data_module.dataset import DatasetBase
from oat.data_module.dataset_helpers import (
    ERA5_DERIVED_FEATURES as DERIVED_FEATURES,
    build_obs_variable_support as _build_obs_variable_support,
    dtype_map,
)
from oat.data_module.field_spec import FieldConfig, parse_field_config
from oat.data_module.meteostat_observations import MeteostatObservations
from oat.data_module.sources import PointSetSource, RasterSource

_LOG = logging.getLogger(__name__)

# Mapping from obs variable name -> (required NWP vars, conversion fn taking a
# name -> [..., N] column dict and returning the obs-unit tensor). Copied from
# legacy.ERA5MeteostatDataset.get_baseline_pre_transform's ``_mapping`` for
# use with DatasetBase.get_baseline_pre_transform.
ERA5_OBS_TO_NWP: dict[str, tuple[list[str], Callable[[dict[str, Tensor]], Tensor]]] = {
    "temp": (["2t"], lambda n: n["2t"] - 273.15),
    "wspd": (["10u", "10v"], lambda n: torch.sqrt(n["10u"] ** 2 + n["10v"] ** 2)),
    "wpgt": (["10u", "10v"], lambda n: torch.sqrt(n["10u"] ** 2 + n["10v"] ** 2)),
    "prcp": (["tp"], lambda n: n["tp"] * 1000.0),
}


class ERA5MeteostatDataset(DatasetBase):
    """``DatasetBase`` wired to the existing ERA5 / Meteostat loaders.

    Accepts the same ``config`` shape as ``legacy.ERA5MeteostatDataset``
    (data paths, variable lists, station/time filters, ``catalog``) plus the field
    schema (``coord_sources``, ``fields``, ``roles``, ``transform_type``). Builds an
    obs ``PointSetSource``, an nwp ``PointSetSource`` (native ERA5 lon/lat
    grid points) and a topo ``RasterSource``, then delegates assembly to
    ``DatasetBase``.

    CRS note: unlike the COSMO field dataset (Swiss LV95 eastings/northings),
    all coordinates here are geographic ``[longitude, latitude]`` in degrees.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.catalog = config.catalog
        obs_var_support = _build_obs_variable_support(
            self.catalog, [str(v) for v in config.obs_vars]
        )
        dtype = dtype_map[str(config.dtype)]

        self._load_nwp(config, dtype)
        self._load_topo(config, dtype)
        self._load_obs(config, dtype)

        field_config = self._parse_field_config(config)
        sources = {
            "obs": self._build_obs_source(),
            "nwp": self._build_nwp_source(),
            "topo": self._build_topo_source(),
        }
        coord_source_coords = self._build_coord_source_coords(field_config)

        super().__init__(
            field_config=field_config,
            coord_source_coords=coord_source_coords,
            sources=sources,
            times=self.times,
            obs_var_support=obs_var_support,
            dtype=dtype,
        )
        self._mark_shared_memory()

    # -- field schema --
    @staticmethod
    def _parse_field_config(config: Any) -> FieldConfig:
        container = (
            OmegaConf.to_container(config, resolve=True)
            if OmegaConf.is_config(config)
            else dict(config)
        )
        return parse_field_config(cast(dict[str, Any], container))

    # -- NWP --
    def _load_nwp(self, config: Any, dtype: torch.dtype) -> None:
        from anemoi.datasets import open_dataset

        self.nwp_ds = open_dataset(config.nwp_path)
        nwp_vars = [str(v) for v in config.nwp_vars]
        source_vars = list(
            dict.fromkeys(
                [v for v in nwp_vars if v not in DERIVED_FEATURES]
                + [
                    dep
                    for v in nwp_vars
                    if v in DERIVED_FEATURES
                    for dep in DERIVED_FEATURES[v][0]
                ]
            )
        )
        missing_vars = [v for v in source_vars if v not in self.nwp_ds.name_to_index]
        if missing_vars:
            raise KeyError(
                f"NWP variables (or their dependencies) not in dataset: "
                f"{missing_vars}. Available: {list(self.nwp_ds.name_to_index.keys())}"
            )
        self._nwp_out_vars = nwp_vars
        self._nwp_source_vars = source_vars
        self.nwp_var_idx = [self.nwp_ds.name_to_index[v] for v in source_vars]

        # ERA5 longitudes are stored in [0, 360); wrap to [-180, 180] so the
        # nwp coord source shares the same lon/lat convention as topo/obs.
        nwp_lons = np.where(
            self.nwp_ds.longitudes > 180,
            self.nwp_ds.longitudes - 360,
            self.nwp_ds.longitudes,
        )
        self.nwp_lonlat = torch.tensor(
            np.column_stack([nwp_lons, self.nwp_ds.latitudes]), dtype=dtype
        )

        self._setup_time_axis(config)

    def _setup_time_axis(self, config: Any) -> None:
        self.obs_ds = MeteostatObservations(
            obs_path=config.obs_path,
            station_meta_path=config.station_meta_path,
        )
        self.stations = (
            np.array(list(config.stations), dtype=str)
            if config.stations is not None
            else self.obs_ds.stations
        )

        all_times = pd.to_datetime(self.nwp_ds.dates).tz_localize("UTC")
        if config.start_date is not None or config.end_date is not None:
            start = (
                pd.to_datetime(config.start_date).tz_localize("UTC")
                if config.start_date
                else all_times.min()
            )
            end = (
                pd.to_datetime(config.end_date).tz_localize("UTC")
                if config.end_date
                else all_times.max()
            )
            mask = (all_times >= start) & (all_times <= end)
            self.times = all_times[mask]
            if len(self.times) == 0:
                raise ValueError(
                    f"No NWP times found within the requested date range "
                    f"({start} to {end}). Available time range is "
                    f"{all_times.min()} to {all_times.max()}."
                )
            self.time_indices = np.where(mask)[0]
        else:
            self.times = all_times
            self.time_indices = np.arange(len(all_times))

        missing_dates = self.nwp_ds.missing
        if missing_dates is not None and len(missing_dates) > 0:
            missing_arr = np.asarray(list(missing_dates))
            missing_idx = np.isin(self.time_indices, missing_arr)
            self.times = self.times[~missing_idx]
            self.time_indices = self.time_indices[~missing_idx]

    def _nwp_values(self, time_idx: int) -> Tensor:
        nwp_idx = int(self.time_indices[time_idx])
        raw = self.nwp_ds[nwp_idx][self.nwp_var_idx]
        raw = raw.reshape(len(self._nwp_source_vars), -1)
        raw_t = torch.from_numpy(raw).to(self.dtype)
        source = {var: raw_t[i] for i, var in enumerate(self._nwp_source_vars)}
        channels = [
            (
                DERIVED_FEATURES[v][1](*[source[d] for d in DERIVED_FEATURES[v][0]])
                if v in DERIVED_FEATURES
                else source[v]
            )
            for v in self._nwp_out_vars
        ]
        return torch.stack(channels, dim=0)

    def _build_nwp_source(self) -> PointSetSource:
        units = {v: str(self.catalog.nwp_vars[v].unit) for v in self._nwp_out_vars}
        return PointSetSource(
            self.nwp_lonlat,
            self._nwp_out_vars,
            values_fn=self._nwp_values,
            units=units,
            native_coord_source="nwp_grid",
        )

    # -- topo --
    def _load_topo(self, config: Any, dtype: torch.dtype) -> None:
        self._topo_vars = [str(v) for v in config.topo_vars]
        topo_ds = xr.open_zarr(config.topo_path)

        # Data vars are (latitude, longitude); RasterSource.build_index does
        # row * W + col with row from y (=latitude), col from x (=longitude),
        # and assumes strictly increasing (origin, res) axes. Flip to
        # ascending latitude first (mirrors legacy.ERA5MeteostatDataset._load_topo)
        # so origin=(lon[0], lat[0]) / res=(dlon, dlat) index correctly; a
        # lat/lon swap or descending-latitude origin here would silently
        # misalign every topo lookup.
        if float(topo_ds.latitude[0]) > float(topo_ds.latitude[-1]):
            topo_ds = topo_ds.isel(latitude=slice(None, None, -1))
        self.topo_ds = topo_ds[self._topo_vars]

        topo_arr = np.stack([self.topo_ds[v].values for v in self._topo_vars], axis=0)
        self.topo_field = torch.from_numpy(np.nan_to_num(topo_arr, nan=0.0)).to(dtype)

    def _build_topo_source(self) -> RasterSource:
        units = {v: str(self.catalog.topo_vars[v].unit) for v in self._topo_vars}
        lon = self.topo_ds.longitude.values
        lat = self.topo_ds.latitude.values
        return RasterSource(
            self.topo_field,
            origin=(float(lon[0]), float(lat[0])),
            res=(float(lon[1] - lon[0]), float(lat[1] - lat[0])),
            var_names=self._topo_vars,
            units=units,
        )

    # -- obs --
    def _load_obs(self, config: Any, dtype: torch.dtype) -> None:
        if self.times is None or len(self.times) == 0:
            raise ValueError(
                "ERA5MeteostatDataset._load_obs called with no valid "
                "times available (self.times is empty)."
            )
        self._obs_vars = [str(v) for v in config.obs_vars]
        obs_arr, mask_arr = self.obs_ds.get_observations(
            first_date=self.times[0],
            last_date=self.times[-1] + pd.Timedelta(hours=1),
            parameters=self._obs_vars,
            stations=self.stations,
            return_mask=True,
            as_numpy=True,
        )
        local_indices = self.time_indices - self.time_indices[0]
        self.obs_tensor = torch.from_numpy(obs_arr[local_indices])
        self.obs_mask_tensor = torch.from_numpy(mask_arr[local_indices])

        for wind_var in ("wspd", "wpgt"):
            if wind_var in self._obs_vars:
                self.obs_tensor[..., self._obs_vars.index(wind_var)] /= 3.6

        coord_vars = [str(v) for v in config.coord_vars]
        meta = self.obs_ds.stations_table.loc[self.stations]
        self.station_coords = torch.tensor(meta[coord_vars].values.copy(), dtype=dtype)
        # Free the big obs DataFrame; obs_tensor is already in shared memory.
        self.obs_ds._obs = None

    def _obs_values(self, t: int) -> Tensor:
        return self.obs_tensor[t].T.nan_to_num(nan=0.0)

    def _obs_mask(self, t: int) -> Tensor:
        return self.obs_mask_tensor[t].T

    def _build_obs_source(self) -> PointSetSource:
        # Note: full spawn-DDP picklability of open zarr/anemoi handles is
        # pre-existing (shared with the old legacy.ERA5MeteostatDataset) and out
        # of scope; this task only removes the newly-introduced lambda
        # unpicklability.
        units = {v: str(self.catalog.obs_vars[v].unit) for v in self._obs_vars}
        return PointSetSource(
            self.station_coords,
            self._obs_vars,
            values_fn=self._obs_values,
            mask_fn=self._obs_mask,
            units=units,
            native_coord_source="stations",
        )

    # -- coord sources --
    def _build_coord_source_coords(
        self, field_config: FieldConfig
    ) -> dict[str, Tensor]:
        return {
            name: (
                self.nwp_lonlat
                if name == field_config.nwp_coord_source
                else self.station_coords
            )
            for name in field_config.coord_sources
        }

    @staticmethod
    def _share(tensor: Any) -> None:
        if tensor is not None:
            tensor.share_memory_()

    def _mark_shared_memory(self) -> None:
        for attr in (
            "topo_field",
            "station_coords",
            "obs_tensor",
            "obs_mask_tensor",
            "nwp_lonlat",
        ):
            self._share(getattr(self, attr, None))
        for idx, w in self._index_cache.values():
            self._share(idx)
            self._share(w)
