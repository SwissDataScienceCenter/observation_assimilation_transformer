from pathlib import Path
import logging
import numpy as np
from oat.data_module import wind_dataset, dataloader, transform
import torch
from omegaconf import DictConfig
import xarray as xr
import pandas as pd
from typing import Any

_NWP_DATA: xr.Dataset | None = None
_OBS_DATA: xr.Dataset | None = None


def _random_split(stations: np.ndarray, n_train: int, n_val: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Split stations randomly into train/val/test sets.

    Args:
        stations: Contains all available stations.
        n_train: Number of training stations
        n_val: Number of validation stations
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_stations, val_stations, test_stations) arrays.
    """
    rng = np.random.default_rng(seed)
    shuffled_stations = rng.permutation(stations)
    train_stations = shuffled_stations[:n_train]
    val_stations = shuffled_stations[n_train:n_train+n_val]
    test_stations = shuffled_stations[n_train+n_val:]
    logging.info(f"Random split: {len(train_stations)} train, {len(val_stations)} val, {len(test_stations)} test stations.")
    return train_stations, val_stations, test_stations


def create_splits(data_cfg: DictConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/val/test splits based on data configuration.
    Args:
        data_cfg: Configuration dictionary for data module.
    Returns:
        Tuple of (train_stations, val_stations, test_stations).
    """

    # --- load available stations ---
    obs_path = Path(data_cfg.paths.obs)
    obs_data = xr.open_zarr(obs_path)
    stations = obs_data.station.to_numpy()

    split_cfg = data_cfg.split
    split_type = split_cfg.type
    
    station_info_path = Path(data_cfg.paths.station_info)
    station_info  = pd.read_csv(station_info_path, index_col=0)    
    
    subset_indices = np.zeros(len(station_info), dtype=bool)
    for subset in split_cfg.subsets:
        # joint subset of all requested subsets
        subset_stations = station_info[subset] == 1
        subset_indices |= subset_stations.to_numpy()
    selected_stations = station_info.loc[subset_indices].index.to_numpy()
    stations = np.intersect1d(stations, selected_stations)
    logging.info(f"Number of stations after applying subsets {split_cfg.subsets}: {len(stations)}")
        
    assert len(stations) > split_cfg.n_train + split_cfg.n_val, \
        f"Not enough stations ({len(stations)}) for requested train ({split_cfg.n_train}) + val ({split_cfg.n_val}) split."

    if split_type == "random":
        return _random_split(
            stations=stations,
            n_train=split_cfg.n_train,
            n_val=split_cfg.n_val,
            seed=split_cfg.seed,
        )
    else:
        raise ValueError(f"Unknown split type: {split_type}")

def get_channel_dims(transform_dict: dict[str, Any]) -> dict[str, Any] | None:
    if transform_dict is None:
        return None
    channel_dims = {}
    for key, val in transform_dict.items():
        if isinstance(val, (dict, DictConfig)):
            channel_dims.update({key: get_channel_dims(val)})
        else:
            for t in val:
                if t.get("channel_dim", None):
                    channel_dims.update({key: t["channel_dim"]})
    return channel_dims


def build_transforms(transform_cfg: DictConfig, stats: dict[str, Any]) -> dict[str, Any]:
    """Build transforms from configuration dictionary.
    Args:
        transform_cfg: Configuration dictionary for transforms.
        stats: Statistics dictionary for normalization transforms.
    Returns:
        Dictionary of transforms keyed by data name.
    """   
    transform_dict = {}
    for key, value in transform_cfg.items():
        assert isinstance(key, str)
        if not isinstance(value, (dict, DictConfig)):
            # Leaf node - build transform list
            transform_dict[key] = _build_transform_list(value, stats.get(key, {}))
        elif isinstance(value, (dict, DictConfig)):
            # Nested structure - recurse
            transform_dict[key] = build_transforms(value, stats.get(key, {}))
        else:
            raise ValueError(f"Unexpected value type at key '{key}': {type(value)}")
    
    return transform_dict


def _build_transform_list(transform_list: list[transform.Transform], transform_stats: dict[str, Any]) -> transform.ComposedTransform:
    """Build a list of transforms from configuration.
    
    Args:
        transform_list: List of transform configuration dictionaries.
        transform_stats: Statistics dictionary for the transforms.
    Returns:
        A composed transform consisting of the specified transforms.
    """
    transforms = []
    for t_cfg in transform_list:
        # assert isinstance(t_cfg, dict)
        transforms.append(_create_transform(t_cfg, transform_stats))
    composed_transform = transform.ComposedTransform(transforms)
    return composed_transform

def _create_transform(t_cfg: dict[str, Any], t_stats: dict[str, Any]) -> transform.Transform:
    """Create a single transform from config.
    
    Args:
        t_cfg: Configuration dictionary for the transform.
        t_stats: Statistics dictionary for the transform.
    Returns:
        An instance of the specified transform.
    """
    transform_type = t_cfg['type']
    params = {k: v for k, v in t_cfg.items() if k != 'type'}
    
    if transform_type == 'standard_normalize':
        params['mean'] = t_stats.get('mean')
        params['std'] = t_stats.get('std')
        return transform.StandardNormalizeTransform(**params)
    elif transform_type == 'log':
        return transform.LogTransform(**params)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


def create_dataset(data_config: DictConfig) -> tuple[dataloader.OATLoader, 
                                                        dataloader.OATLoader, 
                                                        dataloader.OATLoader]:
    """
    Create datasets and dataloaders based on data configuration.
    
    Args:
        data_config: Configuration dictionary for data module.
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Load data configuration
    loader_config = dict(data_config.loader)
    train_stations, val_stations, test_stations = create_splits(data_config)
    nwp_path = Path(data_config.paths.nwp)
    obs_path = Path(data_config.paths.obs)
    topo_path = Path(data_config.paths.topo)
    
    def worker_init_fn(worker_id: int) -> None:
        global _NWP_DATA, _OBS_DATA
        _NWP_DATA = xr.open_zarr(nwp_path, chunks={})  # 0 cost O(1)
        _OBS_DATA = xr.open_zarr(obs_path, chunks={})
        
    nwp_vars = list(data_config.vars.nwp_vars)
    obs_vars = list(data_config.vars.obs_vars)
    topo_vars = list(data_config.vars.topo_vars)

    nwp_data = xr.open_zarr(nwp_path)
    obs_data = xr.open_zarr(obs_path)
    time_set = np.intersect1d(nwp_data.time, obs_data.time)
    
    # create datasets and loaders
    train_ds = wind_dataset.WindDataset(
        times=time_set,
        input_paths={'nwp': nwp_path, 'topo': topo_path},
        input_vars={'nwp': nwp_vars, 'topo': topo_vars},
        target_path=obs_path,
        target_vars=obs_vars,
        stations=train_stations,
        global_grid_reference=data_config.vars.global_grid_reference,
        patch_size=torch.tensor(data_config.vars.get("local_patch_size", (32, 32))),
        dtype=torch.float32,
        use_time_embedding=data_config.use_time_embedding
    )
    val_ds = wind_dataset.WindDataset(
        times=time_set,
        input_paths={'nwp': nwp_path, 'topo': topo_path},
        input_vars={'nwp': nwp_vars, 'topo': topo_vars},
        target_path=obs_path,
        target_vars=obs_vars,
        stations=val_stations,
        global_grid_reference=data_config.vars.global_grid_reference,
        dtype=torch.float32,
        use_time_embedding=data_config.use_time_embedding
    )
    test_ds = wind_dataset.WindDataset(
        times=time_set,
        input_paths={'nwp': nwp_path, 'topo': topo_path},
        input_vars={'nwp': nwp_vars, 'topo': topo_vars},
        target_path=obs_path,
        target_vars=obs_vars,
        stations=test_stations,
        global_grid_reference=data_config.vars.global_grid_reference,
        dtype=torch.float32,
        use_time_embedding=data_config.use_time_embedding
    )
    if not data_config.get("transforms", None):
        return (
            dataloader.OATLoader(
                train_ds,
                shuffle=True,
                worker_init_fn=worker_init_fn,
                **loader_config
            ),
            dataloader.OATLoader(
                val_ds,
                worker_init_fn=worker_init_fn,
                **loader_config
            ),
            dataloader.OATLoader(
                test_ds,
                worker_init_fn=worker_init_fn,
                **loader_config
            )
        )
        
    if data_config.get("pre_stats_transforms", None):
        pre_transforms = None
    else:
        pre_transforms = build_transforms(data_config.pre_stats_transforms, stats={})
        
    stats_loader = dataloader.OATLoader(
        train_ds,
        shuffle=True,
        transforms=pre_transforms,
        worker_init_fn=worker_init_fn,
        **loader_config | {"persistent_workers": False}
    )
    
    chan_dims = get_channel_dims(data_config.transforms)
    
    stats = dataloader.compute_stats_from_loader(stats_loader, 
                                              chan_dims=chan_dims,
                                              max_batches=10, 
                                              device='cpu')

    del stats_loader
    
    transforms = build_transforms(data_config.transforms, stats)
    
    train_loader = dataloader.OATLoader(
        train_ds,
        shuffle=True,
        transforms=transforms,
        worker_init_fn=worker_init_fn,
        **loader_config
    )
    val_loader = dataloader.OATLoader(
        val_ds,
        transforms=transforms,
        worker_init_fn=worker_init_fn,
        **loader_config
    )
    test_loader = dataloader.OATLoader(
        test_ds,
        transforms=transforms,
        worker_init_fn=worker_init_fn,
        **loader_config
    )
    return train_loader, val_loader, test_loader
