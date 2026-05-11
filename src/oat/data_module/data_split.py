import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, Optional, Tuple
from oat.data_module import transform, statistics
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from omegaconf import ListConfig

def create_splits(dataset: Dataset[Dict[str, Any]],
                  logger: logging.Logger | None = None,
                  val_fraction: float = 0.2,
                  test_fraction: float = 0.2,
                  train_split_fraction: float = 0.8,
                  train_indices: Optional[np.ndarray | list| ListConfig] = None,
                  val_indices: Optional[np.ndarray | list| ListConfig] = None,
                  test_indices: Optional[np.ndarray | list| ListConfig] = None,
                  seed: int | None = None) -> Dict[str, Tuple[Dataset, Dataset, Dataset] | 
                                                        Tuple[list[int], list[int], list[int]]]:


    """
    Create train, validation, and test splits from a dataset.
    Splits the dataset by stations into training, validation, and test sets.
    The stations are randomly shuffled before splitting to ensure randomness.
    Args:
        dataset: The input dataset cand split in both the dataset and containing stations to be split.
        logger: Optional logger for logging split information (default: None).
        val_fraction: Fraction of stations to use for validation (default: 0.2).
        test_fraction: Fraction of stations to use for testing (default: 0.2).
        train_split_fraction: Fraction of training stations to use as observed stations (vs query stations) in each training sample (default: 0.8).
        train_indices: optional list of train station indices.
        val_indices: optional list of validation station indices.
        test_indices: optional list of test station indices.
        seed: Optional random seed for reproducibility (default: None).
    Returns:
        A dictionary with "datasets" mapping to a tuple containing the train, validation, and test datasets
        and "station_indices" mapping to a tuple with the corresponding station indices.
    """
    num_stations = len(dataset.stations)
    test_size = int(num_stations * test_fraction)
    val_size = int(num_stations * val_fraction)
    
    if all(x is None for x in (train_indices, val_indices, test_indices)):
        assert val_fraction + test_fraction < 1.0, "Validation and test fractions must sum to less than 1."
        assert val_fraction >= 0.0 and test_fraction >= 0.0, "Validation and test fractions must be non-negative."
        
        indices = np.arange(num_stations, dtype=int)
        rng = torch.Generator().manual_seed(seed) if seed is not None else None
        indices = indices[torch.randperm(num_stations, generator=rng).numpy()]
        
        
        train_indices = indices[:num_stations - (val_size + test_size)]
        val_indices = indices[num_stations - (val_size + test_size):num_stations - test_size]
        test_indices = indices[num_stations - test_size:]
        
        if logger:
            logger.info('Generated new split indices.')
            logger.info(f'Train indices: {train_indices}')
            logger.info(f'Val indices: {val_indices}')
            logger.info(f'Test indices: {test_indices}')
    else:
        assert all(x is not None for x in (train_indices, val_indices, test_indices))
        if logger:
            logger.info("Using split indices defined in the configuration.")
            logger.info(f'Train indices: {train_indices}')
            logger.info(f'Val indices: {val_indices}')
            logger.info(f'Test indices: {test_indices}')
        
        train_indices = np.array(train_indices) if  isinstance(train_indices, (list, ListConfig)) else train_indices
        val_indices = np.array(val_indices) if  isinstance(val_indices, (list, ListConfig)) else val_indices
        test_indices = np.array(test_indices) if  isinstance(test_indices, (list, ListConfig)) else test_indices
        
        if logger:
            if len(val_indices) != val_size:
                logger.warning(f'Length of the val indices list ({len(val_indices)}) does not match val_fraction {val_fraction}. \
                    Make sure this is intentional!')
            if len(test_indices) != test_size:
                logger.warning(f'Length of the test indices list ({len(test_indices)}) does not match val_fraction {test_fraction}. \
                    Make sure this is intentional!')
            if len(train_indices) != (num_stations - val_size - test_size):
                logger.warning(f'Length of the training indices list ({len(train_indices)}) does not match val_fraction {val_fraction} and test_fraction {test_fraction}. \
                    Make sure this is intentional!')
        
    
    assert len(train_indices) > 0 and len(val_indices) > 0 and len(test_indices) > 0, "Each split must contain at least one station."
    
    train_dataset = TrainDataset(dataset, train_indices, split_fraction=train_split_fraction, seed=seed)
    
    observed_indices = np.array([], dtype=int) if train_split_fraction == 0.0 else train_indices
    val_dataset = EvalDataset(dataset, eval_indices=val_indices, observed_indices=observed_indices)
    
    observed_indices = np.array([], dtype=int) if train_split_fraction == 0.0 else np.concatenate((train_indices, val_indices))
    test_dataset = EvalDataset(dataset, eval_indices=test_indices, observed_indices=observed_indices)
    
    return {
        'datasets': (train_dataset, val_dataset, test_dataset),
        'station_indices': (train_indices.tolist(), val_indices.tolist(), test_indices.tolist())
    }

class TrainDataset(Dataset[Dict[str, Any]]):
    """
    A dataset wrapper that splits training station data into observed and query sets.
    This dataset takes a base dataset and a set of training station indices, then
    randomly splits those stations into two groups for each sample: observed stations
    (keys) and query stations. The split is performed dynamically for each access
    to introduce variability during training.
    Args:
        dataset: The underlying dataset to wrap
        train_indices: Array of station indices to use for training
        split_fraction: Fraction of stations to use as observed/key stations (default: 0.8)
        seed: Optional random seed for reproducibility (default: None)
    Attributes:
        dataset: The wrapped dataset
        train_indices: Indices of training stations
        stations: Station data corresponding to train_indices
        split_fraction: Fraction used for observed stations
        rng: Random number generator for permutations
    Returns:
        A dictionary containing:
            - index: The sample index
            - observation_data: Data from observed/key stations
            - query_data: Data from query stations
            - global_data: Global data unchanged from original dataset
    """
    
    def __init__(self,
                 dataset: Dataset[Dict[str, Any]],
                 train_indices: np.ndarray,
                 split_fraction: float = 0.8,
                 seed: int | None = None):
        self.dataset = dataset
        self.train_indices = train_indices
        self.stations = dataset.stations[self.train_indices]
        self.split_fraction = split_fraction
        self.rng = torch.Generator().manual_seed(seed) if seed is not None else None
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    @property
    def station_names(self) -> list[str]:
        """Return list of station names in the training set"""
        return self.stations.tolist()
    
    @property
    def station_info(self) -> pd.DataFrame:
        """Return station metadata for training stations as DataFrame"""
        return self.dataset.obs_ds.stations_table.loc[self.stations]
    
    @property
    def variable_names(self) -> dict[str, Any]:
        """Return list of variable names in the dataset"""
        return self.dataset.variable_names
    
    @property
    def num_stations(self) -> int:
        return len(self.train_indices)
    
    @property
    def num_observed_stations(self) -> int:
        return int(len(self.train_indices) * self.split_fraction)
    
    @property
    def num_query_stations(self) -> int:
        return len(self.train_indices) - self.num_observed_stations
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.dataset[idx]
        
        permuted_indices = self.train_indices[torch.randperm(len(self.train_indices), generator=self.rng).numpy()]
        split_point = self.num_observed_stations
        key_indices = permuted_indices[:split_point]
        query_indices = permuted_indices[split_point:]
        
        observation_data = data['observation_data']
        
        key_data, query_data = {}, {}
        
        for key in observation_data:
            key_data[key] = observation_data[key][key_indices]
            query_data[key] = observation_data[key][query_indices]
            
        key_data['station_index'] = key_indices
        query_data['station_index'] = query_indices
        
        split_data = data | {
            'observation_data': key_data,
            'query_data': query_data,
        }
        return split_data
    
    def get_transform(self, num_data_points: int, dataloader: DataLoader | None=None) -> tuple[transform.NestedTransform, transform.Transform]:
        """Compute data-driven transforms from random samples.

        Uses the variable catalog (if configured) to determine log pre-transforms
        and validate distribution–support compatibility.
        """
        from oat.data_module.variable_catalog import VariableDef, load_catalog, check_distribution_compatibility
        from oat.model_module.output_head import OutputHead

        obs_vars = list(self.dataset.config.obs_vars)
        var_settings: dict[str, Any] = dict(self.dataset.config.var_settings) if self.dataset.config.var_settings else {}
        var_to_dist: dict[str, str] = dict(self.dataset.config.obs_variable_to_distribution)

        catalog: dict[str, VariableDef] = {}
        if self.dataset.config.variable_catalog_path is not None:
            catalog = load_catalog(self.dataset.config.variable_catalog_path)

        # Determine which obs_vars get a log pre-transform
        log_var_indices: list[int] = []
        for i, var in enumerate(obs_vars):
            use_log = var_settings.get(var, {}).get("use_log", False)
            if not use_log:
                continue
            if var not in catalog:
                raise ValueError(
                    f"var_settings['{var}'].use_log=True requires the variable to be in the "
                    f"variable catalog, but '{var}' was not found. Add it to the catalog "
                    f"at variable_catalog_path."
                )
            if catalog[var].support != "positive":
                raise ValueError(
                    f"var_settings['{var}'].use_log=True requires 'positive' support in the "
                    f"variable catalog, but '{var}' has support '{catalog[var].support}'. "
                    f"Only strictly positive variables can be modelled in log space."
                )
            log_var_indices.append(i)

        if log_var_indices:
            log_transform: transform.Transform | None = transform.LogTransform(
                apply_to=log_var_indices, feature_dim=-1, offset=0.0
            )
            pre_transform: transform.NestedTransform | None = transform.NestedTransform(
                {"observation_data": {"obs": log_transform}}
            )
        else:
            log_transform = None
            pre_transform = None

        # Validate distribution–support compatibility for every obs var
        name_to_dist_cls = OutputHead.name_to_dist_cls()
        log_var_names = {obs_vars[i] for i in log_var_indices}
        for var in obs_vars:
            if var not in var_to_dist or var not in catalog:
                continue
            dist_name = var_to_dist[var]
            dist_cls = name_to_dist_cls.get(dist_name)
            if dist_cls is None:
                continue
            effective_support = "real" if var in log_var_names else catalog[var].support
            check_distribution_compatibility(var, effective_support, dist_name, dist_cls)

        # Initialize statistics gatherers
        stats_dict: dict[str, Any] = {
            'observation_data': {
                'obs': statistics.RunningFeatureStats(
                    num_features=len(obs_vars),
                    feature_dim=-1,
                ),
                'topo': statistics.RunningFeatureStats(
                    num_features=len(self.dataset.config.topo_vars),
                    feature_dim=-3,
                ),
                'geo': statistics.RunningFeatureStats(
                    num_features=len(self.dataset.config.geo_vars),
                    feature_dim=-1,
                ),
            },
            'global_data': {
                'global_field': statistics.RunningFeatureStats(
                    num_features=len(self.dataset.config.nwp_vars) + len(self.dataset.config.topo_vars),
                    feature_dim=-3,
                )
            }
        }

        def gather_stats(sample: dict | torch.Tensor, stats_gatherer: dict | statistics.RunningFeatureStats) -> None:
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
                pre_transformed_batch = pre_transform(batch) if pre_transform is not None else batch
                gather_stats(pre_transformed_batch, stats_dict)
        else:
            rand_indices = torch.randperm(len(self))[:num_data_points]
            for idx in tqdm(rand_indices, desc="Gathering dataset statistics"):
                sample = self[int(idx)]
                if pre_transform is not None:
                    sample = pre_transform(sample)
                gather_stats(sample, stats_dict)

        obs_normalization = stats_dict['observation_data']['obs'].get_standard_transform()

        if log_transform is not None:
            output_transform: transform.Transform = transform.ComposedTransform([log_transform, obs_normalization])
        else:
            output_transform = obs_normalization

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
        return post_transform, output_transform
    
    
class EvalDataset(Dataset[Dict[str, Any]]):
    """
    A dataset wrapper that splits observations into evaluation and observed subsets.
    This class wraps an existing dataset and splits its station-based observations into two groups:
    - Observed stations: Used as context/input for the model
    - Evaluation stations: Used as query points for prediction/evaluation
    The class maintains references to both subsets and provides convenience properties for
    accessing station counts. When indexing, it returns a dictionary with separate entries
    for observation_data (observed stations) and query_data (evaluation stations).
    Args:
        dataset: The underlying dataset to wrap, must have a 'stations' attribute
        eval_indices: Array of indices indicating which stations are used for evaluation
        observed_indices: Array of indices indicating which stations are observed/context
    Attributes:
        dataset: The wrapped dataset
        eval_indices: Indices of evaluation stations
        observed_indices: Indices of observed stations
        eval_stations: Station metadata for evaluation subset
        observed_stations: Station metadata for observed subset
    Properties:
        num_stations: Total number of stations (eval + observed)
        num_observed_stations: Number of observed/context stations
        num_query_stations: Number of evaluation/query stations
    Returns:
        A dictionary containing:
            - 'index': The sample index
            - 'observation_data': Data for observed stations
            - 'query_data': Data for evaluation stations
            - 'global_data': Global data from the original sample
    """
    
    def __init__(self,
                 dataset: Dataset[Dict[str, Any]],
                 eval_indices: np.ndarray,
                 observed_indices: np.ndarray):
        self.dataset = dataset
        self.eval_indices = eval_indices
        self.observed_indices = observed_indices
        
        assert hasattr(dataset, "stations"), "Dataset must have stations attribute."
        assert len(dataset.stations) > 0, "Dataset stations attribute must not be empty."
        self.eval_stations = dataset.stations[self.eval_indices]
        self.observed_stations = dataset.stations[self.observed_indices]
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    @property
    def station_names(self) -> dict[str, list[str]]:
        """Return dict with station names for eval and observed subsets"""
        return {
            'eval': self.eval_stations.tolist(),
            'observed': self.observed_stations.tolist()
        }
        
    @property
    def station_info(self) -> dict[str, pd.DataFrame]:
        """Return dict with station metadata DataFrames for eval and observed subsets"""
        return {
            'eval': self.dataset.obs_ds.stations_table.loc[self.eval_stations],
            'observed': self.dataset.obs_ds.stations_table.loc[self.observed_stations]
        }
        
    @property
    def variable_names(self) -> dict[str, Any]:
        """Return variable names from the underlying dataset"""
        return self.dataset.variable_names
    
    @property
    def num_stations(self) -> int:
        return len(self.eval_indices) + len(self.observed_indices)
    
    @property
    def num_observed_stations(self) -> int:
        return len(self.observed_indices)
    
    @property
    def num_query_stations(self) -> int:
        return len(self.eval_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.dataset[idx]
        observation_data = data['observation_data']
        eval_data, observed_data = {}, {}
        
        for key in observation_data:
            eval_data[key] = observation_data[key][self.eval_indices]
            observed_data[key] = observation_data[key][self.observed_indices]
            
        split_data = data | {
            'observation_data': observed_data,
            'query_data': eval_data
        }
        
        return split_data