"""
COSMO-PeakWeather Dataset for Neural Weather Downscaling

Loads coarse NWP fields, high-res topography, and sparse station observations.
Matches architecture from Neural_Downscaling_with_Station_Observation.pdf
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree
from torch.nn.functional import grid_sample
from torch.utils.data import Dataset


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
    
    # Cache
    cache_dir: str = ".cache_constant_inputs"
    clear_cache: bool = False
    
    # Data type
    dtype: torch.dtype = torch.float32


class CosmoPeakWeatherDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    Dataset combining COSMO NWP fields with PeakWeather station observations.
    
    Returns batches with:
        - global_data: NWP fields + topography at coarse resolution
        - observation_data: Station coords, topo patches, observations
        - time_emb: Sin/cos encodings of diurnal/annual cycles
    """
    
    def __init__(self, config: CosmoPeakWeatherConfig):
        self.config = config
        
        # Load datasets
        self._load_datasets()
        
        # Setup spatial grids and transforms
        self._setup_spatial_grids()
        
        # Cache constant inputs (topo)
        self._cache_constant_data()
        
        # Load observations
        self._load_observations()
        
    def _load_datasets(self) -> None:
        """Load NWP, topography, and station data"""
        from anemoi.datasets import open_dataset
        from peakweather.dataset import PeakWeatherDataset
        
        # NWP fields (time-varying)
        self.nwp_ds = open_dataset(self.config.nwp_path)
        # Validate requested NWP variables exist in the dataset
        missing_vars = [
            v for v in self.config.nwp_vars
            if v not in self.nwp_ds.name_to_index
        ]
        if missing_vars:
            available_vars = list(self.nwp_ds.name_to_index.keys())
            raise KeyError(
                f"The following NWP variables are not present in the dataset: "
                f"{missing_vars}. Available variables are: {available_vars}"
            )
        self.nwp_var_idx = [
            self.nwp_ds.name_to_index[v] for v in self.config.nwp_vars
        ]
        
        # Reference grid for spatial alignment
        self.ref_nwp_ds = xr.open_zarr(self.config.ref_nwp_path)
        
        # Topography (constant)
        self.topo_ds = xr.open_zarr(self.config.topo_path)[self.config.topo_vars]
        
        # Station observations
        self.obs_ds = PeakWeatherDataset(
            root=self.config.obs_path,
            pad_missing_values=False,
            station_type='meteo_station'
        )
        
        # Set stations (all if not specified)
        if self.config.stations is None:
            self.stations = self.obs_ds.stations
        else:
            self.stations = self.config.stations
        
        # Time axis
        all_times = pd.to_datetime(self.nwp_ds.dates).tz_localize('UTC')
        
        # Filter by date range if specified
        if self.config.start_date is not None or self.config.end_date is not None:
            start = pd.to_datetime(self.config.start_date).tz_localize('UTC') if self.config.start_date else all_times.min()
            end = pd.to_datetime(self.config.end_date).tz_localize('UTC') if self.config.end_date else all_times.max()
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
        
        # Project topo onto reference NWP grid for global field
        topo_interp = self.topo_ds[self.config.topo_vars].interp(
            x=self.ref_nwp_ds.x,
            y=self.ref_nwp_ds.y,
            method='nearest'
        ).to_array().values
        
        topo_interp = np.nan_to_num(topo_interp, nan=0.0)
        self.topo_on_nwp_grid = torch.from_numpy(topo_interp).to(self.config.dtype)
        
    def _load_observations(self) -> None:
        """Load station observations for entire time period"""
        # Defensive check: ensure there is at least one valid time before indexing
        if not self.times:
            raise ValueError(
                "CosmoPeakWeatherDataset._load_observations called with no valid "
                "times available (self.times is empty). Ensure times are set "
                "before loading observations."
            )
        self.obs_df = self.obs_ds.get_observations(
            first_date=self.times[0],
            last_date=self.times[-1],
            parameters=self.config.obs_vars,
            stations=self.stations,
        )
        
        # Station metadata
        self.station_coords = torch.tensor(
            self.obs_ds.stations_table.loc[self.stations][
                ['swiss_easting', 'swiss_northing']
            ].values,
            dtype=self.config.dtype
        )
        
        self.station_geo = torch.tensor(
            self.obs_ds.stations_table.loc[self.stations][
                ['swiss_easting', 'swiss_northing', 'dem']
            ].values,
            dtype=self.config.dtype
        )
        
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
        nwp_time = self.nwp_ds.dates[nwp_idx]
        
        # === Global data ===
        # Get NWP fields
        values = self.nwp_ds[
            self.nwp_ds.dates_interval_to_indices(
                start=nwp_time, end=nwp_time
            )[0],
            self.nwp_var_idx,
            0,  # member
            :
        ].reshape(len(self.nwp_var_idx), -1)
        
        # Regrid to reference grid
        nwp_field = values[:, self.nwp_indices]
        nwp_field = torch.from_numpy(
            nwp_field.reshape(len(self.nwp_var_idx), *self.nwp_shape)
        ).to(self.config.dtype)
        
        # Concatenate with topo
        global_field = torch.cat([nwp_field, self.topo_on_nwp_grid], dim=0)
        
        global_data = {'global_field': global_field}
        
        if self.config.use_time_embedding:
            global_data['time_emb'] = self._get_time_embedding(nwp_time)
            
        # === Observation data ===
        # Get observations for this time, with explicit error handling
        try:
            df_time = self.obs_df.xs(time)
        except KeyError as exc:
            raise KeyError(
                f"Observations for time {time!r} not found in obs_df index."
            ) from exc
        df_unstacked = df_time.unstack()
        df_final = df_unstacked.reindex(
            columns=self.config.obs_vars,
            index=self.stations
        )
        obs_values = df_final.values  # (N_stations, C_obs)
        
        # Extract topographic patches at station locations
        topo_field = torch.from_numpy(
            self.topo_data[self.topo_var_idx]
        ).to(self.config.dtype)
        
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
            'obs': torch.from_numpy(obs_values).to(self.config.dtype),
            'mask_obs': torch.from_numpy(~np.isnan(obs_values)),
        }
        
        return {
            'index': idx,
            'global_data': global_data,
            'observation_data': observation_data,
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
        ], dtype=self.config.dtype)


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