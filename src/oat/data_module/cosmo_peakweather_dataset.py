"""COSMO-PeakWeather dataset that emits a dict of ``Field`` per sample.

Thin concrete subclass of ``DatasetBase``. It loads the COSMO NWP,
topography and PeakWeather observations. It wires them into three ``Source``
objects (obs, nwp, topo) consumed by the config-driven field assembly in
``DatasetBase``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import xarray as xr
from omegaconf import OmegaConf
from pyproj import Transformer
from torch import Tensor

from oat.data_module.dataset import DatasetBase
from oat.data_module.dataset_helpers import (
    build_obs_variable_support as _build_obs_variable_support,
)
from oat.data_module.dataset_helpers import (
    dtype_map,
)
from oat.data_module.field_spec import FieldConfig, parse_field_config
from oat.data_module.sources import PointSetSource, RasterSource

_LOG = logging.getLogger(__name__)

# (deps, fn) where fn(*dep_tensors) -> tensor (H, W). Wind speed/direction
# derived from the COSMO U/V 10m wind components.
DERIVED_FEATURES: dict[str, tuple[list[str], Callable[..., torch.Tensor]]] = {
    "wind_speed": (["U_10M", "V_10M"], lambda u, v: torch.sqrt(u**2 + v**2)),
    "wind_direction": (
        ["U_10M", "V_10M"],
        lambda u, v: (torch.rad2deg(torch.atan2(u, v)) + 180.0) % 360.0,
    ),
}

# Mapping from obs variable name -> (required NWP vars, conversion fn taking a
# name -> [..., N] column dict and returning the obs-unit tensor). Surfaced on
# the dataset via the obs_to_nwp property for the NWPInterpolator baseline.
COSMO_OBS_TO_NWP: dict[str, tuple[list[str], Callable[[dict[str, Tensor]], Tensor]]] = {
    "temperature": (["T_2M"], lambda n: n["T_2M"] - 273.15),
    "wind_u": (["U_10M"], lambda n: n["U_10M"]),
    "wind_v": (["V_10M"], lambda n: n["V_10M"]),
    "wind_speed": (
        ["U_10M", "V_10M"],
        lambda n: torch.sqrt(n["U_10M"] ** 2 + n["V_10M"] ** 2),
    ),
    # The archive recipe rescales TOT_PREC by 0.001 (kg/m^2 -> m), so
    # TOT_PREC_1H is in m/h. Multiply by 1000 to get mm/h.
    "precipitation": (["TOT_PREC_1H"], lambda n: 1000.0 * n["TOT_PREC_1H"]),
}


class CosmoPeakWeatherDataset(DatasetBase):
    """``DatasetBase`` wired to the existing COSMO / PeakWeather loaders.

    Accepts a ``config`` with data paths, variable lists, station/time filters
    and ``catalog``, plus the field schema
    (``coord_sources``, ``fields``, ``roles``). Builds an obs
    ``PointSetSource``, an nwp ``PointSetSource`` (native grid points in
    EPSG:2056) and a topo ``RasterSource``, then delegates assembly to
    ``DatasetBase``.
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
        coord_source_shapes: dict[str, tuple[int, int]] = {}
        if field_config.nwp_coord_source is not None:
            fs = self.nwp_ds.field_shape
            # Only record a shape when the NWP grid is truly 2D (COSMO is a
            # rectangular grid; other backends may not be, see ERA5).
            if len(fs) == 2:
                coord_source_shapes[field_config.nwp_coord_source] = (
                    int(fs[0]),
                    int(fs[1]),
                )

        super().__init__(
            field_config=field_config,
            coord_source_coords=coord_source_coords,
            sources=sources,
            times=self.times,
            obs_var_support=obs_var_support,
            dtype=dtype,
            coord_source_shapes=coord_source_shapes,
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

    @property
    def obs_to_nwp(
        self,
    ) -> dict[str, tuple[list[str], Callable[[dict[str, Tensor]], Tensor]]]:
        return COSMO_OBS_TO_NWP

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

        nwp_extracted_path = config.nwp_extracted_path
        if nwp_extracted_path and Path(nwp_extracted_path).exists():
            import zarr as _zarr

            self._nwp_store = _zarr.open(nwp_extracted_path, mode="r")
            if "variables" not in self._nwp_store.attrs:
                raise ValueError(
                    f"Extracted NWP store at {nwp_extracted_path} is missing the "
                    f"required attrs['variables'] metadata. Regenerate the store "
                    f"with scripts/extract_nwp_vars.py."
                )
            extracted_vars = list(self._nwp_store.attrs["variables"])
            missing_extracted = [v for v in source_vars if v not in extracted_vars]
            if missing_extracted:
                raise ValueError(
                    f"Extracted NWP store at {nwp_extracted_path} is missing "
                    f"required variables: {missing_extracted}. Available "
                    f"variables: {extracted_vars}. Regenerate the store with "
                    f"scripts/extract_nwp_vars.py."
                )
            self._extracted_var_idx = [extracted_vars.index(v) for v in source_vars]
        else:
            if nwp_extracted_path:
                _LOG.warning(
                    "nwp_extracted_path=%s does not exist; falling back to anemoi "
                    "(slow: full-chunk reads). Run scripts/extract_nwp_vars.py to "
                    "create it.",
                    nwp_extracted_path,
                )
            self._nwp_store = None

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
        eastings, northings = transformer.transform(
            self.nwp_ds.longitudes, self.nwp_ds.latitudes
        )
        self.nwp_xy = torch.tensor(np.column_stack([eastings, northings]), dtype=dtype)

        self._setup_time_axis(config)

    def _setup_time_axis(self, config: Any) -> None:
        from peakweather.dataset import PeakWeatherDataset

        self.obs_ds = PeakWeatherDataset(
            root=config.obs_path,
            station_type=config.station_type,
            freq="1h",
        )
        self.stations = (
            self.obs_ds.stations if config.stations is None else config.stations
        )

        all_times = pd.to_datetime(self.nwp_ds.dates).tz_localize("UTC")
        start_time = max(all_times.min(), self.obs_ds.observations.index.min())
        end_time = min(all_times.max(), self.obs_ds.observations.index.max())
        if config.start_date is not None or config.end_date is not None:
            start = (
                pd.to_datetime(config.start_date).tz_localize("UTC")
                if config.start_date
                else start_time
            )
            end = (
                pd.to_datetime(config.end_date).tz_localize("UTC")
                if config.end_date
                else end_time
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
        if self._nwp_store is not None:
            raw = self._nwp_store[nwp_idx][self._extracted_var_idx]
        else:
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
            self.nwp_xy,
            self._nwp_out_vars,
            values_fn=self._nwp_values,
            units=units,
            native_coord_source="nwp_grid",
        )

    # -- topo --
    def _load_topo(self, config: Any, dtype: torch.dtype) -> None:
        self._topo_vars = [str(v) for v in config.topo_vars]
        self.topo_ds = xr.open_zarr(config.topo_path)[self._topo_vars]

        cache_dir = Path(str(config.cache_dir))
        cache_dir.mkdir(exist_ok=True, parents=True)
        cache_file = cache_dir / "topo.npy"
        if config.clear_cache and cache_file.exists():
            cache_file.unlink()
        if not cache_file.exists():
            arr = self.topo_ds.to_array().values.astype(np.float32)
            np.save(cache_file, arr)
        topo_data = np.load(cache_file, mmap_mode="r")

        topo_data_vars = list(self.topo_ds.data_vars)
        self.topo_var_idx = [
            topo_data_vars.index(var)
            for var in self._topo_vars
            if var in topo_data_vars
        ]
        topo_subset = topo_data[self.topo_var_idx].copy()
        self.topo_field = torch.from_numpy(topo_subset).to(dtype)

        # Kept for build_domain_grid (geo_overlays.py), which clamps plotting
        # grids to the region where topo_field is finite.
        self.topo_extent = {
            "min": torch.tensor(
                [float(self.topo_ds.x.min()), float(self.topo_ds.y.min())]
            ),
            "max": torch.tensor(
                [float(self.topo_ds.x.max()), float(self.topo_ds.y.max())]
            ),
        }

    def _build_topo_source(self) -> RasterSource:
        units = {v: str(self.catalog.topo_vars[v].unit) for v in self._topo_vars}
        x = self.topo_ds.x.values
        y = self.topo_ds.y.values
        return RasterSource(
            self.topo_field,
            origin=(float(x[0]), float(y[0])),
            res=(float(x[1] - x[0]), float(y[1] - y[0])),
            var_names=self._topo_vars,
            units=units,
        )

    # -- obs --
    def _load_obs(self, config: Any, dtype: torch.dtype) -> None:
        if self.times is None or len(self.times) == 0:
            raise ValueError(
                "CosmoPeakWeatherDataset._load_obs called with no valid "
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

        coord_vars = [str(v) for v in config.coord_vars]
        self.station_coords = torch.tensor(
            self.obs_ds.stations_table.loc[self.stations][coord_vars].values,
            dtype=dtype,
        )

    def _obs_values(self, t: int) -> Tensor:
        return self.obs_tensor[t].T.nan_to_num(nan=0.0)

    def _obs_mask(self, t: int) -> Tensor:
        return self.obs_mask_tensor[t].T

    def _build_obs_source(self) -> PointSetSource:
        # Note: full spawn-DDP picklability of open zarr/anemoi handles is
        # pre-existing and out
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
                self.nwp_xy
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
            "nwp_xy",
        ):
            self._share(getattr(self, attr, None))
        for idx, w in self._index_cache.values():
            self._share(idx)
            self._share(w)
