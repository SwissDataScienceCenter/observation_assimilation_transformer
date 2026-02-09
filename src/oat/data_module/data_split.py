import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict
from oat.data_module import transform, statistics
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

def create_splits(dataset: Dataset[Dict[str, Any]],
                  val_fraction: float = 0.2,
                  test_fraction: float = 0.2,
                  train_split_fraction: float = 0.8,
                  seed: int | None = None) -> tuple[Dataset[Dict[str, Any]], 
                                                              Dataset[Dict[str, Any]], 
                                                              Dataset[Dict[str, Any]]]:
    """
    Create train, validation, and test splits from a dataset.
    Splits the dataset by stations into training, validation, and test sets.
    The stations are randomly shuffled before splitting to ensure randomness.
    Args:
        dataset: The input dataset containing stations to be split.
        val_fraction: Fraction of stations to use for validation (default: 0.2).
        test_fraction: Fraction of stations to use for testing (default: 0.2).
        train_split_fraction: Fraction of training stations to use as observed stations (vs query stations) in each training sample (default: 0.8).
        seed: Optional random seed for reproducibility (default: None).
    Returns:
        A tuple containing the train, validation, and test datasets.
    """
    
    assert val_fraction + test_fraction < 1.0, "Validation and test fractions must sum to less than 1."
    assert val_fraction >= 0.0 and test_fraction >= 0.0, "Validation and test fractions must be non-negative."
    
    num_stations = len(dataset.stations)
    indices = np.arange(num_stations, dtype=int)
    rng = torch.Generator().manual_seed(seed) if seed is not None else None
    indices = indices[torch.randperm(num_stations, generator=rng).numpy()]
    
    test_size = int(num_stations * test_fraction)
    val_size = int(num_stations * val_fraction)
    
    train_indices = indices[:num_stations - (val_size + test_size)]
    val_indices = indices[num_stations - (val_size + test_size):num_stations - test_size]
    test_indices = indices[num_stations - test_size:]
    
    assert len(train_indices) > 0 and len(val_indices) > 0 and len(test_indices) > 0, "Each split must contain at least one station."
    
    train_dataset = TrainDataset(dataset, train_indices, split_fraction=train_split_fraction, seed=seed)
    val_dataset = EvalDataset(dataset, val_indices, train_indices)
    test_dataset = EvalDataset(dataset, test_indices, np.concatenate((train_indices, val_indices)))
    
    return train_dataset, val_dataset, test_dataset

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
        """
        Compute data-driven transforms (log, standardization) from random samples.
            num_data_points: Number of random data points to sample for statistics
        Returns: NestedTransform to apply to batches, and output_transform for obs-only.
        """
        apply_log = ['wind_gust', 'wind_speed']
        log_indices = [i for i, var in enumerate(self.dataset.config.obs_vars) if var in apply_log]

        # Create pre-transform with log
        log_transform = transform.LogTransform(apply_to=log_indices, 
                                               feature_dim=-1, 
                                               offset=1)
        pre_transform = transform.NestedTransform({
            'observation_data': {
                'obs': log_transform
            }
        })
        
        # Initialize statistics gatherers
        stats_dict = {
            'observation_data': {
                'obs': statistics.RunningFeatureStats(
                    num_features=len(self.dataset.config.obs_vars),
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
        # Gather statistics
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
                pre_transformed_batch = pre_transform(batch)
                gather_stats(pre_transformed_batch, stats_dict)
        else:
            rand_indices = torch.randperm(len(self))[:num_data_points]
            for idx in tqdm(rand_indices, desc="Gathering dataset statistics"):
                sample = self[int(idx)]
                pre_transformed_sample = pre_transform(sample)
                gather_stats(pre_transformed_sample, stats_dict)
        
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
            
        eval_data['station_index'] = self.eval_indices
        observed_data['station_index'] = self.observed_indices
        eval_data['stations_names'] = self.dataset.stations[self.eval_indices].to_numpy()
        observed_data['stations_names'] = self.dataset.stations[self.observed_indices].to_numpy()
        
        split_data = data | {
            'observation_data': observed_data,
            'query_data': eval_data
        }
        
        return split_data