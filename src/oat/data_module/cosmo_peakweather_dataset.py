"""
COSMO-PeakWeather Dataset for Neural Weather Downscaling

Loads coarse NWP fields, high-res topography, and sparse station observations.
Matches architecture from Neural_Downscaling_with_Station_Observation.pdf
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree
from torch.nn.functional import grid_sample
from torch.utils.data import Dataset, DataLoader
from oat.data_module import transform, statistics
from oat.data_module.support import VariableSupport
import logging as _logging
from tqdm import tqdm

dtype_map = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# (deps, fn) where fn(*dep_tensors) → tensor (H, W)
DERIVED_FEATURES: dict[str, tuple[list[str], Callable[..., torch.Tensor]]] = {
    'wind_speed': (['U_10M', 'V_10M'], lambda u, v: torch.sqrt(u**2 + v**2)),
    'wind_direction': (['U_10M', 'V_10M'], lambda u, v: (torch.rad2deg(torch.atan2(u, v)) + 180.0) % 360.0),
}

# Mapping from obs variable name → (list of required NWP vars, conversion fn).
# fn receives a dict {nwp_var: tensor(H, W)} and returns tensor(H, W).
_OBS_NWP_MAPPING: dict[str, tuple[list[str], Callable[..., torch.Tensor]]] = {
    'temperature':   (['T_2M'],             lambda nwp: nwp['T_2M'] - 273.15),
    'wind_u':        (['U_10M'],            lambda nwp: nwp['U_10M']),
    'wind_v':        (['V_10M'],            lambda nwp: nwp['V_10M']),
    'wind_speed':    (['U_10M', 'V_10M'],   lambda nwp: torch.sqrt(nwp['U_10M'] ** 2 + nwp['V_10M'] ** 2)),
    'precipitation': (['TOT_PREC_1H'],      lambda nwp: 600.0 * nwp['TOT_PREC_1H']),
}


def _build_obs_variable_support(catalog: Any, obs_vars: list[str]) -> VariableSupport:
    catalog_obs_vars = catalog.obs_vars
    missing_vars = [var for var in obs_vars if var not in catalog_obs_vars]
    if missing_vars:
        raise KeyError(f"Observation variables missing from catalog: {missing_vars}")

    lower_bounds = torch.tensor(
        [catalog_obs_vars[var].get("lower_bound", float("-inf")) for var in obs_vars],
        dtype=torch.float32,
    )
    upper_bounds = torch.tensor(
        [catalog_obs_vars[var].get("upper_bound", float("inf")) for var in obs_vars],
        dtype=torch.float32,
    )
    lower_open = torch.tensor(
        [catalog_obs_vars[var].get("lower_open", False) for var in obs_vars],
        dtype=torch.bool,
    )
    upper_open = torch.tensor(
        [catalog_obs_vars[var].get("upper_open", False) for var in obs_vars],
        dtype=torch.bool,
    )

    return VariableSupport(
        lower_bounds,
        upper_bounds,
        lower_open=lower_open,
        upper_open=upper_open,
    )


@dataclass
class CosmoPeakWeatherConfig:
    """Configuration for COSMO-PeakWeather dataset"""
    # Data paths
    nwp_path: str
    topo_path: str
    obs_path: str
    ref_nwp_path: str  # Reference grid for regridding

    # Variables
    nwp_vars: list[str]
    topo_vars: list[str]
    obs_vars: list[str]
    
    # Stations (if None, uses all available stations)
    stations: Optional[list[str]] = None
    
    # Time range (if None, uses full available range)
    start_date: Optional[str] = None  # ISO format, e.g. '2020-01-01'
    end_date: Optional[str] = None    # ISO format, e.g. '2023-12-31'
    
    # Topographic patch settings
    patch_size: tuple[int, int] = (32, 32)
    
    # Time embedding
    use_time_embedding: bool = True
    
    # Extracted NWP zarr (rechunked for fast __getitem__)
    nwp_extracted_path: Optional[str] = None  # if set, used instead of anemoi for __getitem__ reads (105x less I/O)

    # Cache
    cache_dir: str = ".cache_constant_inputs"
    clear_cache: bool = False
    
    # Data type
    dtype: str = "float32"
    obs_plot_config: Any = None


class CosmoPeakWeatherDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    Dataset combining COSMO NWP fields with PeakWeather station observations.
    
    Returns batches with:
        - global_data: NWP fields + topography at coarse resolution
        - observation_data: Station coords, topo patches, observations
        - time_emb: Sin/cos encodings of diurnal/annual cycles
    """
    
    def __init__(self, config: Any):
        self.config = config
        self.catalog = self.config.catalog
        self.obs_var_support = _build_obs_variable_support(self.catalog, self.config.obs_vars)
        
        # Load datasets
        self._load_datasets()
        
        # Setup spatial grids and transforms
        self._setup_spatial_grids()
        
        # Cache constant inputs (topo)
        self._cache_constant_data()
        
        # Load observations
        self._load_observations()
        
        self._mark_shared_memory()
        
    @property
    def variable_names(self) -> dict[str, dict[str, Optional[list[str]]]]:
        return {
            'global_data': {
                'global_field': self.config.nwp_vars + self.config.topo_vars,
                'time_emb': ['sin_day', 'cos_day', 'sin_year', 'cos_year'] if self.config.use_time_embedding else None,
            },
            'observation_data': {
                'coords': self.config.coord_vars,
                'geo': self.config.geo_vars,
                'topo': self.config.topo_vars,
                'obs': self.config.obs_vars,
                'mask_obs': self.config.obs_vars, 
            }
        }

    @property
    def station_names(self) -> list[str]:
        """Return list of station names in the dataset"""
        return self.stations.tolist()
    
    @property
    def station_info(self) -> pd.DataFrame:
        """Return station metadata as DataFrame"""
        return self.obs_ds.stations_table.loc[self.stations]
        
    def _mark_shared_memory(self) -> None:
        """Mark tensors for sharing across workers"""
        for attr in (
            "topo_field", "topo_on_nwp_grid", "station_coords", "station_geo",
            "obs_tensor", "obs_mask_tensor",
        ):
            if hasattr(self, attr):
                getattr(self, attr).share_memory_()
        
    def _load_datasets(self) -> None:
        """Load NWP, topography, and station data"""
        from anemoi.datasets import open_dataset
        from peakweather.dataset import PeakWeatherDataset
        
        # NWP fields (time-varying)
        self.nwp_ds = open_dataset(self.config.nwp_path)
        # Collect all source vars: base vars + deps of derived vars (deduplicated)
        source_vars = list(dict.fromkeys(
            [v for v in self.config.nwp_vars if v not in DERIVED_FEATURES]
            + [dep for v in self.config.nwp_vars if v in DERIVED_FEATURES
               for dep in DERIVED_FEATURES[v][0]]
        ))
        missing_vars = [v for v in source_vars if v not in self.nwp_ds.name_to_index]
        if missing_vars:
            raise KeyError(f"NWP variables (or their dependencies) not in dataset: {missing_vars}. "
                           f"Available: {list(self.nwp_ds.name_to_index.keys())}")
        self._source_vars = source_vars
        self.nwp_var_idx = [self.nwp_ds.name_to_index[v] for v in source_vars]

        if self.config.nwp_extracted_path and Path(self.config.nwp_extracted_path).exists():
            import zarr as _zarr
            self._nwp_store = _zarr.open(self.config.nwp_extracted_path, mode="r")
            extracted_vars = list(self._nwp_store.attrs["variables"])
            if "variables" not in self._nwp_store.attrs:
                raise ValueError(
                    f"Extracted NWP store at {self.config.nwp_extracted_path} is missing the "
                    f"required attrs['variables'] metadata. Regenerate the store with "
                    f"scripts/extract_nwp_vars.py."
                )
            extracted_vars = list(self._nwp_store.attrs["variables"])
            missing_extracted_vars = [v for v in self._source_vars if v not in extracted_vars]
            if missing_extracted_vars:
                raise ValueError(
                    f"Extracted NWP store at {self.config.nwp_extracted_path} is missing "
                    f"required variables: {missing_extracted_vars}. Available variables: "
                    f"{extracted_vars}. Regenerate the store with scripts/extract_nwp_vars.py."
                )
            self._extracted_var_idx = [extracted_vars.index(v) for v in self._source_vars]
            _logging.getLogger(__name__).info(
                "Using extracted NWP store at %s with variables: %s",
                self.config.nwp_extracted_path,
                extracted_vars
            )
        else:
            if self.config.nwp_extracted_path:
                _logging.getLogger(__name__).warning(
                    "nwp_extracted_path=%s does not exist; falling back to anemoi (slow: 379 MB/read). "
                    "Run scripts/extract_nwp_vars.py to create it.",
                    self.config.nwp_extracted_path,
                )
            self._nwp_store = None

        # Reference grid for spatial alignment
        self.ref_nwp_ds = xr.open_zarr(self.config.ref_nwp_path)
        topo_vars = [str(var) for var in self.config.topo_vars]
        # Topography (constant)
        self.topo_ds = xr.open_zarr(self.config.topo_path)[topo_vars]
        
        # Station observations
        self.obs_ds = PeakWeatherDataset(
            root=self.config.obs_path,
            station_type=self.config.station_type,
            freq="1h",
        )
        
        # Set stations (all if not specified)
        if self.config.stations is None:
            self.stations = self.obs_ds.stations
        else:
            self.stations = self.config.stations
        
        # Time axis
        all_times = pd.to_datetime(self.nwp_ds.dates).tz_localize('UTC')
        start_time = max(all_times.min(), self.obs_ds.observations.index.min())
        end_time = min(all_times.max(), self.obs_ds.observations.index.max())
        # Filter by date range if specified
        if self.config.start_date is not None or self.config.end_date is not None:
            start = pd.to_datetime(self.config.start_date).tz_localize('UTC') if self.config.start_date else start_time
            end = pd.to_datetime(self.config.end_date).tz_localize('UTC') if self.config.end_date else end_time
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
            missing_arr = np.asarray([n for n in missing_dates])
            missing_idx = np.isin(self.time_indices, missing_arr)
            self.times = self.times[~missing_idx]
            self.time_indices = self.time_indices[~missing_idx]
        
    def _setup_spatial_grids(self) -> None:
        """Setup coordinate systems and KDTree for NWP regridding"""
        # Transformer: lat/lon → Swiss coordinates (EPSG:2056)
        self.transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:2056", always_xy=True
        )
        
        # NWP coordinates in Swiss system
        lon, lat = self.nwp_ds.longitudes, self.nwp_ds.latitudes
        eastings, northings = self.transformer.transform(lon, lat)
        
        # KDTree for nearest-neighbor regridding
        points = np.column_stack([eastings, northings])
        self.nwp_tree = cKDTree(points)
        
        # Reference grid
        grid_e, grid_n = np.meshgrid(
            self.ref_nwp_ds.x.values,
            self.ref_nwp_ds.y.values
        )
        self.nwp_shape = grid_e.shape
        
        # Query indices for regridding
        grid_points = np.column_stack([grid_e.ravel(), grid_n.ravel()])
        _, self.nwp_indices = self.nwp_tree.query(grid_points)
        
        # Spatial extents for grid_sample
        self.topo_extent = {
            'min': torch.tensor([
                float(self.topo_ds.x.min()),
                float(self.topo_ds.y.min())
            ]),
            'max': torch.tensor([
                float(self.topo_ds.x.max()),
                float(self.topo_ds.y.max())
            ])
        }
        
        self.nwp_extent = {
            'min': torch.tensor([
                float(self.ref_nwp_ds.x.min()),
                float(self.ref_nwp_ds.y.min())
            ]),
            'max': torch.tensor([
                float(self.ref_nwp_ds.x.max()),
                float(self.ref_nwp_ds.y.max())
            ])
        }
        
        # Resolution for patch extraction
        topo_res = float(self.topo_ds.x[1] - self.topo_ds.x[0])
        nwp_res = float(self.ref_nwp_ds.x[1] - self.ref_nwp_ds.x[0])
        
        # Setup grids for grid_sample
        self.topo_grid = self._create_patch_grid(
            self.config.patch_size, topo_res
        )
        self.nwp_grid = self._create_patch_grid(
            self.config.patch_size, nwp_res
        )
        
    def _create_patch_grid(
        self, 
        patch_size: tuple[int, int], 
        resolution: float
    ) -> torch.Tensor:
        """Create grid for extracting local patches via grid_sample"""
        h, w = patch_size
        half_h = (h // 2) * resolution
        half_w = (w // 2) * resolution
        
        x = torch.linspace(-half_h, half_h, h)
        y = torch.linspace(-half_w, half_w, w)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        
        return torch.stack([grid_x, grid_y], dim=-1)  # (h, w, 2)
        
    def _cache_constant_data(self) -> None:
        """Cache topography as memmap for efficient access"""
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / "topo.npy"
        
        # Clear cache if requested
        if self.config.clear_cache and cache_file.exists():
            cache_file.unlink()
        
        if cache_file.exists():
            self.topo_data = np.load(cache_file, mmap_mode='r')
        else:
            # Load and save
            arr = self.topo_ds.to_array().values.astype(np.float32)
            np.save(cache_file, arr)
            self.topo_data = np.load(cache_file, mmap_mode='r')
            
        # Get indices of variables we want, preserving config.topo_vars order
        topo_data_vars = list(self.topo_ds.data_vars)
        self.topo_var_idx = [
            topo_data_vars.index(var)
            for var in self.config.topo_vars
            if var in topo_data_vars
        ]
        topo_vars = [str(var) for var in self.config.topo_vars]
        
        # Project topo onto reference NWP grid for global field
        topo_interp = self.topo_ds[topo_vars].interp(
            x=self.ref_nwp_ds.x,
            y=self.ref_nwp_ds.y,
            method='nearest'
        ).to_array().values
        
        topo_interp = np.nan_to_num(topo_interp, nan=0.0)
        self.topo_on_nwp_grid = torch.from_numpy(topo_interp).to(dtype_map[self.config.dtype])
        topo_subset = self.topo_data[self.topo_var_idx].copy()
        self.topo_field = torch.from_numpy(topo_subset).to(
            dtype_map[self.config.dtype]
        )
        
    def _load_observations(self) -> None:
        """Load station observations for entire time period"""
        # Defensive check: ensure there is at least one valid time before indexing
        if self.times is None or len(self.times) == 0:
            raise ValueError(
                "CosmoPeakWeatherDataset._load_observations called with no valid "
                "times available (self.times is empty). Ensure times are set "
                "before loading observations."
            )
        obs_arr, mask_arr = self.obs_ds.get_observations(
            first_date=self.times[0],
            last_date=self.times[-1] + pd.Timedelta(hours=1),  # Include last time
            parameters=self.config.obs_vars,
            stations=self.stations,
            return_mask=True,
            as_numpy=True,
        )
        # Align with filtered times: convert absolute NWP positions to local offsets
        # from the fetched obs array (which starts at self.times[0]).
        local_indices = self.time_indices - self.time_indices[0]
        self.obs_tensor = torch.from_numpy(obs_arr[local_indices])
        self.obs_mask_tensor = torch.from_numpy(mask_arr[local_indices])
        
        # Station metadata
        self.station_coords = torch.tensor(
            self.obs_ds.stations_table.loc[self.stations][
                self.config.coord_vars
            ].values,
            dtype=dtype_map[self.config.dtype]
        )
        
        self.station_geo = self._get_geo(self.station_coords)

        
    def __len__(self) -> int:
        return len(self.times)
        
    def __getitem__(self, idx: int) -> dict:
        """
        Returns batch matching OAT architecture:
        
        global_data:
            - global_field: (C_nwp + C_topo, H, W)
            - time_emb: (4,) if use_time_embedding
            
        observation_data:
            - coords: (N_stations, 2)
            - geo: (N_stations, 3)  # easting, northing, elevation
            - topo: (N_stations, C_topo, patch_h, patch_w)
            - obs: (N_stations, C_obs)
            - mask_obs: (N_stations, C_obs)
        """
        time = self.times[idx]
        nwp_idx = self.time_indices[idx]

        # === Global data ===
        # Get NWP fields
        if self._nwp_store is not None:
            raw = self._nwp_store[int(nwp_idx)][self._extracted_var_idx]  # (n_vars, n_grid)
        else:
            raw = self.nwp_ds[int(nwp_idx)][self.nwp_var_idx]             # fallback: reads full 379 MB chunk
        raw = raw.reshape(len(self.nwp_var_idx), -1)
        raw_field = torch.from_numpy(
            raw[:, self.nwp_indices].reshape(len(self.nwp_var_idx), *self.nwp_shape)
        ).to(dtype_map[self.config.dtype])
        source = {var: raw_field[i] for i, var in enumerate(self._source_vars)}

        # Output channels in exactly the order of self.config.nwp_vars
        nwp_channels = [
            DERIVED_FEATURES[v][1](*[source[d] for d in DERIVED_FEATURES[v][0]])
            if v in DERIVED_FEATURES else source[v]
            for v in self.config.nwp_vars
        ]
        nwp_field = torch.stack(nwp_channels, dim=0)  # (len(nwp_vars), H, W)
        global_field = torch.cat([nwp_field, self.topo_on_nwp_grid], dim=0)

        global_data = {'global_field': global_field}

        if self.config.use_time_embedding:
            global_data['time_emb'] = self._get_time_embedding(time)
            
        obs_values = self.obs_tensor[idx]       # (N_stations, C_obs)
        obs_mask = self.obs_mask_tensor[idx]    # (N_stations, C_obs)

        topo_field = self.topo_field  # No loading, no conversion
        topo_patches = self._extract_patches(
            self.station_coords,
            topo_field,
            self.topo_grid,
            self.topo_extent
        )
        
        observation_data = {
            'coords': self.station_coords,
            'geo': self.station_geo,
            'topo': topo_patches,
            'obs': obs_values.to(dtype_map[self.config.dtype]).nan_to_num(nan=0.0),
            'mask_obs': obs_mask.to(dtype_map[self.config.dtype]),
        }
        
        return {
            'index': idx,
            'global_data': global_data,
            'observation_data': observation_data,
        }
        
    def get_query(self, query_coords: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get query data (topo, geo) for given station coordinates"""
        topo_query = self._extract_patches(
            query_coords,
            self.topo_field,
            self.topo_grid,
            self.topo_extent
        )
        
        geo_query = self._get_geo(query_coords)
        
        return {
            'coords': query_coords,
            'topo': topo_query,
            'geo': geo_query
        }
        
    def _extract_patches(
        self,
        positions: torch.Tensor,
        field: torch.Tensor,
        grid: torch.Tensor,
        extent: dict
    ) -> torch.Tensor:
        """Extract local patches around positions using grid_sample"""
        # Transform to [-1, 1] for grid_sample
        grid_pos = positions.unsqueeze(1).unsqueeze(1) + grid
        grid_scaled = self._normalize_positions(
            grid_pos, extent['min'], extent['max']
        )
        
        # Sample
        patches = grid_sample(
            field.unsqueeze(0).expand(positions.shape[0], -1, -1, -1),
            grid_scaled,
            align_corners=True,
            mode='bilinear',
            padding_mode='border',
        )
        
        return patches
        
    def _get_geo(self, coords: torch.Tensor) -> torch.Tensor:
        """Return geo features (swiss_easting, swiss_northing, dem) for given coordinates.

        DEM (topo_vars[0]) is bilinearly interpolated from topo_field at each coordinate.

        Args:
            coords: (N, 2) tensor of [swiss_easting, swiss_northing]

        Returns:
            (N, 3) tensor of [swiss_easting, swiss_northing, dem]
        """
        point_grid = torch.zeros(1, 1, 2, dtype=coords.dtype, device=coords.device)
        dem = self._extract_patches(
            coords,
            self.topo_field[0:1],  # DEM channel only, shape (1, H, W)
            point_grid,
            self.topo_extent,
        )[:, 0, 0, 0]  # (N,)
        return torch.cat([coords, dem.unsqueeze(1)], dim=1)  # (N, 3)

    @staticmethod
    def _normalize_positions(
        pos: torch.Tensor,
        min_val: torch.Tensor,
        max_val: torch.Tensor
    ) -> torch.Tensor:
        """Transform positions to [-1, 1] for grid_sample"""
        return 2 * (pos - min_val) / (max_val - min_val) - 1
        
    def _get_time_embedding(self, datetime: pd.Timestamp) -> torch.Tensor:
        """
        Sin/cos encoding of diurnal and annual cycles.
        Returns: (4,) tensor [sin_day, cos_day, sin_year, cos_year]
        """
        dt = pd.to_datetime(datetime)
        
        # Time of day
        seconds_in_day = 86400
        seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        day_angle = 2 * np.pi * seconds / seconds_in_day
        
        # Time of year
        year_length = 365.2425  # Average year length accounting for leap years
        doy = dt.dayofyear + seconds / seconds_in_day
        year_angle = 2 * np.pi * doy / year_length
        
        return torch.tensor([
            np.sin(day_angle),
            np.cos(day_angle),
            np.sin(year_angle),
            np.cos(year_angle)
        ], dtype=dtype_map[self.config.dtype])
        
    def get_transform(self, num_data_points: int, dataloader: DataLoader | None=None, type: str = 'real',
                      data_indices: np.ndarray | None=None) -> tuple[transform.NestedTransform, transform.Transform]:
        """
        Compute data-driven transforms (log, standardization) from random samples.
            num_data_points: Number of random data points to sample for statistics
        Returns: NestedTransform to apply to batches, and output_transform for obs-only.
        """
        
                    
        def _collect_stats(pre_transform: transform.NestedTransform, dataloader: DataLoader | None=None, num_data_points: int | None=None) -> dict[str, dict[str, statistics.RunningFeatureStats]]:
            
            if num_data_points is None:
                num_data_points = len(self)
                
            stats_dict = {
                'observation_data': {
                    'obs': statistics.RunningFeatureStats(
                        num_features=len(self.config.obs_vars),
                        feature_dim=-1,
                    ),
                    'topo': statistics.RunningFeatureStats(
                        num_features=len(self.config.topo_vars),
                        feature_dim=-3,
                    ),
                    'geo': statistics.RunningFeatureStats(
                        num_features=self.station_geo.shape[-1],
                        feature_dim=-1,
                    ),
                },
                'global_data': {
                    'global_field': statistics.RunningFeatureStats(
                        num_features=len(self.config.nwp_vars) + len(self.config.topo_vars),
                        feature_dim=-3,
                    )
                
                }
            }
            
            def gather_stats(sample: dict[str, Any] | torch.Tensor, stats_gatherer: dict[str, Any] | statistics.RunningFeatureStats) -> None:
                if isinstance(stats_gatherer, statistics.RunningFeatureStats):
                    stats_gatherer.update(sample)
                elif isinstance(stats_gatherer, dict):
                    for key, stat in stats_gatherer.items():
                        gather_stats(sample[key], stat)
            
            if dataloader is not None:
                bs = dataloader.batch_size
                for i, batch in tqdm(enumerate(dataloader), total=num_data_points // bs, desc="Gathering dataset statistics"):
                    if i * bs >= num_data_points:
                        break
                    pre_transformed_batch = pre_transform(batch)
                    gather_stats(pre_transformed_batch, stats_dict)
            else:
                if data_indices is not None:
                    rand_indices = torch.randperm(len(data_indices))[:num_data_points]
                    rand_indices = data_indices[rand_indices]
                else:
                    rand_indices = torch.randperm(len(self))[:num_data_points]
                    
                for idx in tqdm(rand_indices, desc="Gathering dataset statistics"):
                    sample = self[int(idx)]
                    pre_transformed_sample = pre_transform(sample)
                    gather_stats(pre_transformed_sample, stats_dict)
            return stats_dict
            
        offset = 1e-3  # Small offset to avoid log(0)
        
        if type == 'real':
            # Example variables that may benefit from log-transform include non-negative,
            # unbounded quantities such as wind_gust, wind_speed, and precipitation.
            # Select variables with non-negative support and no finite upper bound to help
            # with training stability while preserving the intended support constraints.
            log_indices = [i for i, var in enumerate(self.config.obs_vars) if self.obs_var_support.lower_bounds[i] >= 0.0 and self.obs_var_support.upper_bounds[i] == float('inf')]
            # Create pre-transform with log
            log_transform = transform.LogTransform(apply_to=log_indices, 
                                                feature_dim=-1, 
                                                offset=offset)
            pre_transform = transform.NestedTransform({
                'observation_data': {
                    'obs': log_transform
                }
            })
             # Gather statistics

            stats_dict = _collect_stats(pre_transform, dataloader=dataloader, num_data_points=num_data_points)  # Run once to initialize stats with log transform applied
            output_transform =  transform.ComposedTransform([log_transform, stats_dict['observation_data']['obs'].get_standard_transform()])
            transform_dict = {
                'observation_data': {
                    'obs':output_transform,
                    'topo': stats_dict['observation_data']['topo'].get_standard_transform(),
                    'geo': stats_dict['observation_data']['geo'].get_standard_transform(),
                },
                'query_data': {
                    'obs':output_transform,
                    'topo': stats_dict['observation_data']['topo'].get_standard_transform(),
                    'geo': stats_dict['observation_data']['geo'].get_standard_transform(),
                },
                'global_data': {
                    'global_field': stats_dict['global_data']['global_field'].get_standard_transform(),
                }
            }
            post_transform = transform.NestedTransform(transform_dict)
        elif type == 'keep_support':
            apply_offset = ['wind_gust', 'wind_speed']
            
            # apply only scale that have zero lower bound, to preserve support of variables with non-zero lower bound (e.g. precipitation)
            
            
            offset_indices = [i for i, var in enumerate(self.config.obs_vars) if var in apply_offset]
            offset_arr = torch.zeros(len(self.config.obs_vars))
            offset_arr[offset_indices] = offset
            scale_arr = torch.ones(len(self.config.obs_vars))
            offset_transform = transform.TranslateScaleTransform(
                translate=offset_arr,
                scale=scale_arr,
                channel_dim=-1
            )
            pre_transform = transform.NestedTransform({
                'observation_data': {
                    'obs': offset_transform
                }            
            })
            stats_dict = _collect_stats(pre_transform, dataloader=dataloader, num_data_points=num_data_points)  # Run once to initialize stats with
             # Example variables to standardize; adjust as needed
            
            translate_arr = -stats_dict['observation_data']['obs'].mean.clone()
            # apply only scale where the variable support has zero lower bound and infinite upper bound, to preserve support of variables with finite bounds (e.g. precipitation)
            apply_only_scale_indices = [i for i, var in enumerate(self.config.obs_vars) 
                                        if self.obs_var_support.lower_bounds[i] == 0.0 and self.obs_var_support.upper_bounds[i] == float('inf')]
            translate_arr[apply_only_scale_indices] = 0.0  # Don't translate variables that are only scaled
            scale_arr = 1.0 / stats_dict['observation_data']['obs'].std.clone()
            
            output_transform = transform.ComposedTransform([offset_transform, transform.TranslateScaleTransform(
                translate=translate_arr,
                scale=scale_arr,
                channel_dim=-1
            )])
            transform_dict = {
                'observation_data': {
                    'obs': output_transform,
                    'topo': stats_dict['observation_data']['topo'].get_standard_transform(),
                    'geo': stats_dict['observation_data']['geo'].get_standard_transform(),
                },
                'query_data': {
                    'obs': output_transform,
                    'topo': stats_dict['observation_data']['topo'].get_standard_transform(),
                    'geo': stats_dict['observation_data']['geo'].get_standard_transform(),
                },
                'global_data': {
                    'global_field': stats_dict['global_data']['global_field'].get_standard_transform(),
                }
            }
            post_transform = transform.NestedTransform(transform_dict)
        else:
            raise ValueError(f"Unsupported transform type: {type}. Supported types are 'real' and 'keep_support'.")
            
        return post_transform, output_transform

    def get_baseline_pre_transform(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Return a closure that replaces global_field NWP channels with obs-unit channels.

        Channel order in the returned field matches self.config.obs_vars exactly.
        Raises ValueError if any obs var has no mapping or if a required NWP dep is absent.
        """
        for obs_var in self.config.obs_vars:
            if obs_var not in _OBS_NWP_MAPPING:
                raise ValueError(
                    f"No NWP mapping defined for obs variable {obs_var!r}. "
                    f"Supported: {list(_OBS_NWP_MAPPING)}."
                )
            deps, _ = _OBS_NWP_MAPPING[obs_var]
            missing = [d for d in deps if d not in self.config.nwp_vars]
            if missing:
                raise ValueError(
                    f"NWP deps {missing} for obs variable {obs_var!r} are not in "
                    f"config.nwp_vars {list(self.config.nwp_vars)}."
                )

        nwp_vars = list(self.config.nwp_vars)
        obs_vars = list(self.config.obs_vars)

        def _pre(sample: dict[str, Any]) -> dict[str, Any]:
            nwp_fields = sample['global_data']['global_field']  # (C_nwp + C_topo, H, W)
            nwp = {var: nwp_fields[..., i, :, :] for i, var in enumerate(nwp_vars)}
            new_field = torch.stack([_OBS_NWP_MAPPING[v][1](nwp) for v in obs_vars], dim=-3)
            return {**sample, 'global_data': {**sample['global_data'], 'global_field': new_field}}

        return _pre
# Example usage
if __name__ == "__main__":
    config = CosmoPeakWeatherConfig(
        nwp_path="/scratch/mch/fzanetta/data/anemoi/datasets/mch-co1e-an-archive-0p01-2019-2024-1h-v1-pl13.zarr",
        topo_path="/scratch/mch/cdonner/data/obs_downscaling/raw/cosmo-1e_rechunked/topo.zarr",
        obs_path="/scratch/mch/cdonner/data/peakweather",
        ref_nwp_path="/scratch/mch/cdonner/data/obs_downscaling/raw/cosmo-1e_rechunked/nwp.zarr",
        nwp_vars=['T_2M', 'U_10M', 'V_10M'],
        topo_vars=["DEM", "TPI_500M", "TPI_2000M", "WE_DERIVATIVE_2000M_SIGRATIO1", "SN_DERIVATIVE_2000M_SIGRATIO1"],
        obs_vars=['temperature', 'wind_gust', 'wind_u', 'wind_v'],
        stations=['ABO', 'AEG', 'WYN', 'ZER', 'KLO'],
    )
    
    dataset = CosmoPeakWeatherDataset(config)
    batch = dataset[10]
    
    print("Global field:", batch['global_data']['global_field'].shape)
    print("Station coords:", batch['observation_data']['coords'].shape)
    print("Station geo:", batch['observation_data']['geo'].shape)
    print("Station topo:", batch['observation_data']['topo'].shape)
    print("Observations:", batch['observation_data']['obs'].shape)