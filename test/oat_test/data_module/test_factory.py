import os
import pytest
from omegaconf import OmegaConf
from oat.data_module import factory
import torch
from torch.utils.data import DataLoader

@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipped on CI (requires local datasets)"
)
def test_with_local_data() -> None:
    data_config = OmegaConf.load('configs/dataset/windgust.yaml')
    dataset = factory.create_dataset(data_config)
    loader = DataLoader(
        dataset,
        batch_size=data_config.loader.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    sample_batch = next(iter(loader))
    assert 'global_data' in sample_batch
    # assert 'query_data' in sample_batch
    assert 'observation_data' in sample_batch
    assert 'index' in sample_batch
    
    
    batch_size = data_config.loader.batch_size
    num_chans_global = len(data_config.vars.topo_vars) + len(data_config.vars.nwp_vars)
    
    # check global data shape
    assert sample_batch['global_data']['global_field'].shape[0] == batch_size
    assert sample_batch['global_data']['global_field'].shape[1] == num_chans_global
    assert len(sample_batch['global_data']['global_field'].shape) == 4
    # no nan
    assert not torch.isnan(sample_batch['global_data']['global_field']).any()
    assert len(sample_batch['index']) == batch_size
    # check time_emb
    if data_config.use_time_embedding:
        assert 'time_emb' in sample_batch['global_data']
        assert sample_batch['global_data']['time_emb'].shape[0] == batch_size
        assert len(sample_batch['global_data']['time_emb'].shape) == 2  # default time emb size
        
    # check observation shape
    num_obs = sample_batch['observation_data']['coords'].shape[1]
    for key, data in sample_batch['observation_data'].items():
        if torch.is_tensor(data):
            assert data.shape[0] == batch_size
            assert data.shape[1] == num_obs
            if key == 'coords':
                assert data.shape[2] == 2
            elif key == 'geo':
                # TODO: Define what should be in geo
                pass
            elif key == 'topo':
                assert data.shape[2] == len(data_config.vars.topo_vars)
                assert tuple(data.shape[3:]) == tuple(data_config.vars.local_patch_size)
            elif key == 'obs' or key == 'obs_mask':
                assert data.shape[2] == len(data_config.vars.obs_vars)
                
            
    idx = 0    
    query = dataset.get_query(query_coords=sample_batch['observation_data']['coords'][idx])
    print(query['topo'].shape, sample_batch['observation_data']['topo'][idx].shape)
    assert 'coords' in query
    assert 'topo' in query
    assert 'geo' in query
    assert torch.allclose(query['coords'], sample_batch['observation_data']['coords'][idx])
    assert torch.allclose(query['geo'], sample_batch['observation_data']['geo'][idx])
    assert torch.allclose(query['topo'], sample_batch['observation_data']['topo'][idx])
    