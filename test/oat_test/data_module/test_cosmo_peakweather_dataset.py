from hydra import initialize, compose
from oat.data_module.cosmo_peakweather_dataset import CosmoPeakWeatherDataset, DERIVED_FEATURES
from oat.data_module.transform import NestedTransform, Transform
import numpy as np
import pandas as pd
import pytest
import torch
import os

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


def test_derived_features_wind_speed() -> None:
    u = torch.tensor([[3.0, 0.0]])
    v = torch.tensor([[4.0, 5.0]])
    torch.testing.assert_close(DERIVED_FEATURES['wind_speed'][1](u, v),
                               torch.tensor([[5.0, 5.0]]))

def test_derived_feature_source_vars() -> None:
    """Deps are resolved and deduplicated; base vars before deps."""
    nwp_vars = ['T_2M', 'wind_speed']
    source_vars = list(dict.fromkeys(
        [v for v in nwp_vars if v not in DERIVED_FEATURES]
        + [dep for v in nwp_vars if v in DERIVED_FEATURES for dep in DERIVED_FEATURES[v][0]]
    ))
    assert source_vars == ['T_2M', 'U_10M', 'V_10M']

def test_derived_feature_output_order() -> None:
    """Output channels are in nwp_vars order; unused deps not in output."""
    nwp_vars = ['wind_speed', 'T_2M']
    source = {
        'T_2M': torch.full((2, 2), 280.0),
        'U_10M': torch.full((2, 2), 3.0),
        'V_10M': torch.full((2, 2), 4.0),
    }
    channels = [
        DERIVED_FEATURES[v][1](*[source[d] for d in DERIVED_FEATURES[v][0]])
        if v in DERIVED_FEATURES else source[v]
        for v in nwp_vars
    ]
    result = torch.stack(channels, dim=0)
    assert result.shape[0] == 2
    torch.testing.assert_close(result[0], torch.full((2, 2), 5.0))   # wind_speed
    torch.testing.assert_close(result[1], torch.full((2, 2), 280.0)) # T_2M
