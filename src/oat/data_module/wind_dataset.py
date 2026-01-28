import pandas as pd
from pathlib import Path
import torch 
import numpy as np
import xarray as xr
from torch.nn.functional import grid_sample
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Callable, Tuple, TypedDict, NotRequired
from torch.utils.data.dataloader import _BaseDataLoaderIter
from oat.data_module import transform

import warnings
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

def pos_transform(val: torch.Tensor, min: torch.Tensor, max: torch.Tensor) -> torch.Tensor:
    """Transform positions from high-res to low-res grid coordinates."""
    assert val.shape[-1] == min.shape[-1] == max.shape[-1], "Dimension mismatch in position transformation."
    val_scaled = 2 * (val - min) / (max - min) - 1  # Scale to [-1, 1]
    return val_scaled

def setup_grid(patch_size: torch.Tensor, resolution: dict[str, float]) -> dict[str, torch.Tensor]:
    """Setup grid for grid_sample."""
    grid = {}
    for key, res in resolution.items():
        half_size = (patch_size // 2) * res
        x = torch.linspace(-float(half_size[0]), float(half_size[0]), steps=int(patch_size[0]))
        y = torch.linspace(-float(half_size[1]), float(half_size[1]), steps=int(patch_size[1]))
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        grid[key] = torch.stack((grid_x, grid_y), dim=-1)  # Shape: (patch_size, patch_size, 2)
        
    return grid

def get_context_data(pos: torch.Tensor, grid_data: torch.Tensor, grid: torch.Tensor, grid_extent: dict[str, torch.Tensor]) -> torch.Tensor:
    """Get context data for given positions."""
    grid_pos = pos_transform(grid + pos.unsqueeze(1).unsqueeze(1), grid_extent['min'], grid_extent['max'])
    resampled = grid_sample(grid_data.unsqueeze(0).expand(pos.shape[0], -1, -1, -1), grid_pos, align_corners=True, mode='bilinear')
    return resampled

def get_time_embedding(datetimes: np.ndarray) -> np.ndarray:
    """
    datetimes: array of dtype datetime64[ns] or list-like
    returns: ndarray of shape (N, 4) with:
             [sin_day, cos_day, sin_year, cos_year]
    """

    # Convert to pandas datetime for easy extraction
    dt = pd.to_datetime(datetimes)
    
    # ----------------------------------------------------------
    # 1. Time of day (seconds since midnight)
    # ----------------------------------------------------------
    seconds_in_day = 24 * 3600
    seconds = (
        dt.hour * 3600 +
        dt.minute * 60 +
        dt.second +
        dt.microsecond * 1e-6
    )
    day_angle = 2 * np.pi * seconds / seconds_in_day
    
    sin_day = np.sin(day_angle)
    cos_day = np.cos(day_angle)

    # ----------------------------------------------------------
    # 2. Time of year (day of year + fractional day)
    # ----------------------------------------------------------
    # day of year (1–365/366)
    doy = dt.dayofyear + seconds / seconds_in_day
    year_length = 365.2425  # tropical year

    year_angle = 2 * np.pi * doy / year_length

    sin_year = np.sin(year_angle)
    cos_year = np.cos(year_angle)

    # Stack into final embedding
    return np.stack([sin_day, cos_day, sin_year, cos_year], axis=1)


def _load_constant_as_memmap(key: str, dataset: xr.Dataset, clear_cache: bool = False) -> np.memmap:
    """
    Load a constant (non-time-dependent) dataset as a memory-mapped numpy array.
    If a memmap file does not exist yet, create it.
    """
    cache_dir = Path(".cache_constant_inputs")
    cache_dir.mkdir(exist_ok=True)

    cache_file = cache_dir / f"{key}.npy"

    if clear_cache and cache_file.exists():
        cache_file.unlink()

    if cache_file.exists():
        # Load as memmap
        arr = np.load(cache_file, mmap_mode="r")
        assert isinstance(arr, np.memmap), "Failed to load memmap."
        return arr

    # Otherwise create and save it
    arr = dataset.to_array().values.astype(np.float32)
    np.save(cache_file, arr)          # writes contiguous .npy
    arr_mm = np.load(cache_file, mmap_mode="r")
    assert isinstance(arr_mm, np.memmap), "Failed to create memmap."
    return arr_mm

class QueryBatch(TypedDict):
    coords: torch.Tensor
    topo: torch.Tensor
    geo: torch.Tensor
    obs: NotRequired[torch.Tensor]      # (N, C_obs)
    mask_obs: NotRequired[torch.Tensor] # (N, C_obs)
    
class ObservationBatch(TypedDict):
    coords: torch.Tensor
    geo: torch.Tensor
    obs: torch.Tensor
    mask_obs: torch.Tensor

class OATBatch(TypedDict):
    index: int
    query_data: NotRequired[QueryBatch]
    observation_data: ObservationBatch
    global_data: Dict[str, torch.Tensor]


class WindDataset(Dataset[OATBatch]):
    """Custom Dataset for observation-based downscaling.    """
    
    def __init__(self, 
                times: np.ndarray,
                input_paths: dict[str, Path],
                input_vars: dict[str, list[str]],
                target_path: Path,
                target_vars: list[str],
                global_grid_reference: str,
                stations: list[str] | np.ndarray | None = None,
                patch_size: torch.Tensor = torch.tensor([32, 32]),
                resolution: float | None = None,
                dtype: torch.dtype = torch.float32,
                use_time_embedding: bool = False,
                clear_cache: bool = False,
                station_split_mode: str = 'random',
                seed: int = 42
                ):
        self.dtype = dtype
        self.times = times
        self.target_path = target_path
        self.target_vars = target_vars
        self.stations = stations
        self.vars = {}
        self.input_vars = input_vars
        self._cache: dict[int, dict[str, xr.Dataset]] = {}
        self._check_target_data()
        self.constant_input: list[str] = []
        self.input_extent: dict[str, dict[str, torch.Tensor]] = {}
        self.resolution: dict[str, float] = {}
        self.input_paths = input_paths
        self._check_input_data(resolution=resolution, clear_cache=clear_cache)
        self.global_grid_reference = global_grid_reference
        self.constant_ref = self._project_constant_input_on_ref()
        self.patch_size = torch.as_tensor(patch_size, dtype=torch.int32)
        self.input_grids = setup_grid(self.patch_size, self.resolution)
        self.use_time_embedding = use_time_embedding
        if self.use_time_embedding:
            self.vars['time_embedding'] = ['sin_hour_of_day', 
                                           'cos_hour_of_day', 
                                           'sin_day_of_year', 
                                           'cos_day_of_year']
        self.seed = seed
        self.station_split_mode = station_split_mode
        
    def __len__(self) -> int:
        return len(self.times)

    def _check_input_data(self, resolution: float | None = None, clear_cache: bool = False) -> None:
        self.vars['X'] = []
        self.constant_data_cache = {} # Cache for constant input data
        self.constant_var_indices = {}
        for key, datapath in self.input_paths.items():
            dataset = xr.open_zarr(datapath)[self.input_vars[key]]
            assert 'x' in dataset.dims and 'y' in dataset.dims, f"Dataset {key} must have 'x' and 'y' dimensions."
            self.input_extent[key] = {'min': torch.tensor([torch.tensor(dataset.x.min().values), 
                                                           torch.tensor(dataset.y.min().values)], dtype=self.dtype),
                                      'max': torch.tensor([torch.tensor(dataset.x.max().values), 
                                                           torch.tensor(dataset.y.max().values)], dtype=self.dtype)
                                      }
            inferred_resolution = (dataset.x[1] - dataset.x[0]).values
            if resolution is None:
                self.resolution[key] = inferred_resolution
            else:
                self.resolution[key] = resolution
            if 'time' in dataset.dims:
                # assert 'time' in self.target_data.dims, f"Target data must have 'time' dimension for dataset {key}."
                assert np.all(np.isin(self.times, dataset.time)), \
                    f"Time dimension in dataset {key} does not match target data."
                #self.input_data[key] = dataset.chunk({'time': 1, 'x': 'auto', 'y': 'auto'})
            else:
                self.constant_input.append(key)
                self.constant_data_cache[key] = _load_constant_as_memmap(key, dataset, clear_cache=clear_cache)
                self.constant_var_indices[key] = [idx for idx, var in enumerate(dataset.data_vars) if var in self.input_vars[key]]
            self.vars['X'] += list(self.input_vars[key])
            
    def _project_constant_input_on_ref(self) -> torch.Tensor:
        ref_zarr = xr.open_zarr(self.input_paths[self.global_grid_reference])
        constants = []
        for key in self.constant_input:
            const_data = xr.open_zarr(self.input_paths[key])[self.input_vars[key]]  # (C, H, W)
            interp_data = const_data.interp(x=ref_zarr.x, y=ref_zarr.y, method="nearest").to_array().values  # (C, H_ref, W_ref)
            # set nan to 0 (TODO: better handling?)
            interp_data = np.nan_to_num(interp_data, nan=0.0)
            constants.append(torch.tensor(interp_data, dtype=self.dtype))
        return torch.cat(constants, dim=0)
                
    def _check_target_data(self) -> None:
        target_data = xr.open_zarr(self.target_path)[self.target_vars]
        assert 'station' in target_data.dims, "Target data must have 'station' dimension."
        assert 'time' in target_data.dims, "Target data must have 'time' dimension."
        assert 'x' in target_data.coords and 'y' in target_data.coords, "Target data must have 'x' and 'y' dimensions."
        self.vars['Y'] = list(target_data.data_vars)
        
    def get_query(
        self,
        idx: torch.Tensor,
        query_coords: torch.Tensor,
        # input_data: dict[str, xr.Dataset],
        query_obs: torch.Tensor | None = None
    ) -> QueryBatch:
        time = self.times[idx]
        datasets = self._open_per_worker()
        if datasets is None:
            datasets = self._load_data_lazily()
        input_data = datasets['X']
        topo_q = []
        for key, dataset in input_data.items():
            if key not in self.constant_input:
                data = dataset.sel(time=time).load()
                data_tensor = torch.tensor(data.to_array().values, dtype=self.dtype)
                topo_q.append(get_context_data(query_coords, data_tensor, self.input_grids[key], self.input_extent[key]))
            else:
                data_tensor = torch.from_numpy(self.constant_data_cache[key][self.constant_var_indices[key]]).to(self.dtype)
                topo_q.append(get_context_data(query_coords, data_tensor, self.input_grids[key], self.input_extent[key]))
                      
        query: QueryBatch = {
            'coords': query_coords,
            'topo': torch.cat(topo_q, dim=1),
            'geo': query_coords, # TODO: add elevation later
            'obs': query_obs,
            'mask_obs': ~torch.isnan(query_obs) if query_obs is not None else None
        }
        return query
    
    
    # def get_input_data(self, 
    #             idx: torch.Tensor,
    #             # time: torch.Tensor, 
    #             #query_coords: torch.Tensor,
    #             input_data: dict[str, xr.Dataset], 
    #             # query_obs: torch.Tensor | None = None,
    #             obs_coords: torch.Tensor | None = None,
    #             obs_value_y: torch.Tensor | None = None) -> OATBatch:
        
    #     topo_obs = []
    #     U_global_list = []
    #     for key, dataset in input_data.items():
    #         if key not in self.constant_input:
    #             data = dataset.sel(time=time).load()
    #             data_tensor = torch.tensor(data.to_array().values, dtype=self.dtype)
    #             U_global_list.append(data_tensor)
    #         else:
    #             data_tensor = torch.from_numpy(self.constant_data_cache[key][self.constant_var_indices[key]]).to(self.dtype)
    #             # topo_q.append(get_context_data(query_coords, data_tensor, self.input_grids[key], self.input_extent[key]))
    #             if obs_coords is not None:
    #                 topo_obs.append(get_context_data(obs_coords, data_tensor, self.input_grids[key], self.input_extent[key]))
                      
    #     U_global = torch.cat(U_global_list + [self.constant_ref], dim=0)
    #     global_data = {'global_field': U_global}
    #     if self.use_time_embedding:
    #         global_data['time_emb'] = torch.tensor(get_time_embedding(np.array([time])), dtype=self.dtype).squeeze(0)
            
    #     # query: QueryBatch = {
    #     #     'coords': query_coords,
    #     #     'topo': torch.cat(topo_q, dim=1),
    #     #     'geo': query_coords, # TODO: add elevation later
    #     #     'obs': query_obs,
    #     #     'mask_obs': ~torch.isnan(query_obs) if query_obs is not None else None
    #     # }
    #     observation_data: ObservationBatch = {
    #         'coords': obs_coords,
    #         'topo': torch.cat(topo_obs, dim=1),
    #         'geo': obs_coords, # TODO: add elevation later
    #         'obs': obs_value_y,
    #         'mask_obs': ~torch.isnan(obs_value_y)
    #     }
    #     out: OATBatch = {
    #         'time': time,
    #         'observation_data': observation_data,
    #         'global_data': global_data
    #     }
    #     return out
    
    def _load_data_lazily(self) -> dict[str, xr.Dataset | dict[str, Any]]:
        """Load input datasets lazily."""
        datasets: dict[str, xr.Dataset | dict[str, Any]] = {'X': {}}
        for key, path in self.input_paths.items():
            if key not in self.constant_input:
                datasets['X'][key] = xr.open_zarr(path, consolidated=True)[self.input_vars[key]]
            else:
                datasets['X'][key] = None
        if self.stations is not None:
            datasets['Y'] = xr.open_zarr(self.target_path, consolidated=True)[self.target_vars].sel(station=self.stations)
        else:
            datasets['Y'] = xr.open_zarr(self.target_path, consolidated=True)[self.target_vars]
        return datasets
    
    def _open_per_worker(self) -> dict[str, xr.Dataset] | None:
        info = torch.utils.data.get_worker_info()
        if info is None:
            return None  # main process
        wid = info.id
        if wid not in self._cache:
            self._cache[wid] = self._load_data_lazily()
        return self._cache[wid]
    
    def _split_station_indices(self, total_stations: int, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed + idx)
        indices = np.arange(total_stations)
        # TODO: More modes and more flexible splits
        if self.station_split_mode == 'random':
            rng.shuffle(indices)
            split = total_stations // 2
            return indices[:split], indices[split:]
        else:
            raise ValueError(f"Unknown station_split_mode: {self.station_split_mode}")
    
    def __getitem__(self, idx: int) -> OATBatch:
        time = self.times[idx]
        datasets = self._open_per_worker()
        if datasets is None:
            datasets = self._load_data_lazily()
        obs_dat = datasets['Y'].sel(time=time).load()
        station_pos = torch.tensor(np.asarray([obs_dat.x.values, obs_dat.y.values]), dtype=self.dtype).T
        Y = torch.tensor(obs_dat.to_array().values, dtype=self.dtype).T
        
        input_data = datasets['X']
        topo_obs = []
        U_global_list = []
        
        for key, dataset in input_data.items():
            if key not in self.constant_input:
                data = dataset.sel(time=time).load()
                data_tensor = torch.tensor(data.to_array().values, dtype=self.dtype)
                U_global_list.append(data_tensor)
            else:
                data_tensor = torch.from_numpy(self.constant_data_cache[key][self.constant_var_indices[key]]).to(self.dtype)
                # topo_q.append(get_context_data(query_coords, data_tensor, self.input_grids[key], self.input_extent[key]))
                if station_pos is not None:
                    topo_obs.append(get_context_data(station_pos, data_tensor, self.input_grids[key], self.input_extent[key]))
                      
        U_global = torch.cat(U_global_list + [self.constant_ref], dim=0)
        global_data = {'global_field': U_global}
        if self.use_time_embedding:
            global_data['time_emb'] = torch.tensor(get_time_embedding(np.array([time])), dtype=self.dtype).squeeze(0)
            
        # query: QueryBatch = {
        #     'coords': query_coords,
        #     'topo': torch.cat(topo_q, dim=1),
        #     'geo': query_coords, # TODO: add elevation later
        #     'obs': query_obs,
        #     'mask_obs': ~torch.isnan(query_obs) if query_obs is not None else None
        # }
        observation_data: ObservationBatch = {
            'coords': station_pos,
            'topo': torch.cat(topo_obs, dim=1),
            'geo': station_pos, # TODO: add elevation later
            'obs': Y,
            'mask_obs': ~torch.isnan(Y)
        }
        out: OATBatch = {
            'index': idx,
            'observation_data': observation_data,
            'global_data': global_data
        }
        return out
        # station_pos = torch.tensor(np.asarray([obs_dat.x.values, obs_dat.y.values]), dtype=self.dtype).T
        # Y = torch.tensor(obs_dat.to_array().values, dtype=self.dtype).T
        # # q_idx, kv_idx = self._split_station_indices(station_pos.shape[0], idx)
        # # query_coords = station_pos[q_idx]
        # # key_value_coords = station_pos[kv_idx]
        # # query_y = Y[q_idx]
        # # key_value_y = Y[kv_idx]
        # X = self.get_input_data(time, 
        #                         #query_coords, 
        #                         datasets['X'], 
        #                         # query_obs=query_y, 
        #                         obs_coords=station_pos, 
        #                         obs_value_y=Y)
        # return X
    
    
    # def get_region_query(self,
    #                  time: np.datetime64,
    #                  center_pos: torch.Tensor,
    #                  region_extent: torch.Tensor,
    #                  resolution: float) -> dict[str, torch.Tensor | Tuple[int, int]]:
    #     """
    #     Query a rectangular region around center_pos at given time.
        
    #     center_pos: (2,) tensor with (x, y) coordinates
    #     region_size: (2,) tensor with size of region in each dimension
    #     """
    #     datasets = self._open_per_worker()
    #     if datasets is None:
    #         datasets = self._load_data_lazily()
    #     num_points_x = int(region_extent[0] / resolution)
    #     num_points_y = int(region_extent[1] / resolution)
    #     grid = setup_grid(torch.tensor([num_points_x, num_points_y]), {'query_pos': resolution})['query_pos']
    #     pos = center_pos.unsqueeze(0)  # (1, 2)
    #     query_coord = (grid + pos.unsqueeze(1).unsqueeze(1)).reshape(-1, 2)  # (num_points_x * num_points_y, 2)
    #     query = self.get_input_data(time, query_coord, datasets['X'])
    #     return query | {'pos': query_coord, 'shape': (num_points_x, num_points_y)}