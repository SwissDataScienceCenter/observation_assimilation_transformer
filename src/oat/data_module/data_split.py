import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict

def create_splits(dataset: Dataset[Dict[str, Any]],
                  val_fraction: float = 0.2,
                  test_fraction: float = 0.2,
                  train_split_fraction: float = 0.8) -> tuple[Dataset[Dict[str, Any]], 
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
        train_split_fraction: Fraction of time series data to use for training within the training set (default: 0.8).
    Returns:
        A tuple containing the train, validation, and test datasets.
    """
    num_stations = len(dataset.stations)
    indices = np.arange(num_stations, dtype=int)
    indices = indices[torch.randperm(num_stations)]
    
    test_size = int(num_stations * test_fraction)
    val_size = int(num_stations * val_fraction)
    
    train_indices = indices[:num_stations - (val_size + test_size)]
    val_indices = indices[num_stations - (val_size + test_size):num_stations - test_size]
    test_indices = indices[num_stations - test_size:]
    
    train_dataset = TrainDataset(dataset, train_indices, split_fraction=train_split_fraction)
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
        seed: Random seed for reproducible station splits (default: 42)
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
                 split_fraction: float = 0.8):
        self.dataset = dataset
        self.train_indices = train_indices
        self.stations = dataset.stations[self.train_indices]
        self.split_fraction = split_fraction
        self.rng = torch.Generator()
        
    def __len__(self) -> int:
        return len(self.dataset)
    
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
        
        permuted_indices = self.train_indices[torch.randperm(len(self.train_indices), generator=self.rng)]
        split_point = self.num_observed_stations
        key_indices = permuted_indices[:split_point]
        query_indices = permuted_indices[split_point:]
        
        observation_data = data['observation_data']
        
        key_data, query_data = {}, {}
        
        for key in observation_data:
            key_data[key] = observation_data[key][key_indices]
            query_data[key] = observation_data[key][query_indices]
            
        split_data = {
            'index': idx,
            'observation_data': key_data,
            'query_data': query_data,
            'global_data': data['global_data'],
            'key_indices': key_indices,
            'query_indices': query_indices
        }
        return split_data
    
    
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
            
        split_data = {
            'index': idx,
            'observation_data': observed_data,
            'query_data': eval_data,
            'global_data': data['global_data']
        }
        return split_data