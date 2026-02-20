from hydra import initialize, compose
from oat.data_module.cosmo_peakweather_dataset import CosmoPeakWeatherDataset
from oat.data_module.transform import NestedTransform, Transform
from oat.data_module.data_split import create_splits
import numpy as np
import pandas as pd
import pytest
import os
import torch

@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipped on CI (requires local datasets)"
)
def test_cosmo_peakweather_dataset() -> None:
    with initialize(config_path="../../../configs/", version_base=None):
        cfg = compose(config_name="config")
    
    dataset = CosmoPeakWeatherDataset(cfg.dataset)
    
    # check dates
    # all obs times there
    obs_times = dataset.obs_df.index
    assert np.all(dataset.times.isin(obs_times))
    # only missing nwp times are missing
    nwp_times = pd.to_datetime(dataset.nwp_ds.dates).tz_localize('UTC')
    missing_nwp_idx = np.where(~nwp_times.isin(dataset.times))
    # all missing are in missing
    missing_anemoi_idx = np.asarray([n for n in dataset.nwp_ds.missing])
    assert np.all(np.isin(missing_nwp_idx, missing_anemoi_idx))
    # all missing anemoi are missing
    assert np.all(~np.isin(missing_anemoi_idx, dataset.time_indices))
    
    data_sample = dataset[0]
    assert 'global_data' in data_sample
    assert 'observation_data' in data_sample
    
    var_names = dataset.variable_names
    assert len(data_sample['global_data']['global_field'].shape) == 3
    assert data_sample['global_data']['global_field'].shape[0] == len(var_names['global_data']['global_field'])
    assert len(data_sample['observation_data']['obs'].shape) == 2
    assert data_sample['observation_data']['obs'].shape[-1] == len(var_names['observation_data']['obs'])
    assert data_sample['observation_data']['obs'].shape[0] == len(dataset.stations)
    assert data_sample['observation_data']['mask_obs'].shape == data_sample['observation_data']['obs'].shape
    assert len(data_sample['observation_data']['coords'].shape) == 2
    assert data_sample['observation_data']['coords'].shape[0] == len(dataset.stations)
    assert data_sample['observation_data']['coords'].shape[1] == len(var_names['observation_data']['coords'])
    assert len(data_sample['observation_data']['geo'].shape) == 2
    assert data_sample['observation_data']['geo'].shape[0] == len(dataset.stations)
    assert data_sample['observation_data']['geo'].shape[1] == len(var_names['observation_data']['geo'])
    assert 'time_emb' in data_sample['global_data'] or not dataset.config.use_time_embedding
    assert data_sample['global_data']['time_emb'].shape[0] == len(var_names['global_data']['time_emb']) if dataset.config.use_time_embedding else True
    
    
    
    transforms = dataset.get_transform(0)
    
    assert isinstance(transforms[0], NestedTransform)
    assert isinstance(transforms[1], Transform)
    
@pytest.mark.parametrize(
    "observation_fraction",
    [0.8, 0.0],
    ids=["with_observations", "no_observations"]
)
def test_splits(observation_fraction: float):
    with initialize(config_path="../../../configs/", version_base=None):
        cfg = compose(config_name="config")
    
    dataset = CosmoPeakWeatherDataset(cfg.dataset)
    
    split_cfg = {'val_fraction': 0.2, 
                 'test_fraction': 0.2, 
                 'train_split_fraction': observation_fraction}

    train_dataset, val_dataset, test_dataset = create_splits(dataset, **split_cfg)
    
    num_train_station = int(len(dataset.stations) * (1.0 - split_cfg['val_fraction'] - split_cfg['test_fraction']))
    num_train_and_val_station = int(len(dataset.stations) * (1.0 - split_cfg['test_fraction']))
    
    
    if observation_fraction == 0.0:
        # Assert no observations are returned in the batch
        for ds in [train_dataset, val_dataset, test_dataset]:  
            for v in ds[0]['observation_data'].values():
                assert torch.Tensor(v).numel() == 0
                
        # assert instead that all training stations are used as query
        for v in train_dataset[0]['query_data'].values():
            assert v.shape[0] == num_train_station

    if observation_fraction > 0.0:
        num_key_stations = int(num_train_station * observation_fraction)
        
        # assert the num of observation stations
        for v in train_dataset[0]['observation_data'].values():
                assert v.shape[0] == num_key_stations
    
        for v in val_dataset[0]['observation_data'].values():
                assert v.shape[0] == num_train_station
                
        for v in test_dataset[0]['observation_data'].values():
                assert v.shape[0] == num_train_and_val_station
    
    # TODO check all other shapes (eg query)