import os
import pytest
from omegaconf import OmegaConf
from oat.data_module import factory
import torch



@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipped on CI (requires local datasets)"
)
def test_with_local_data() -> None:
    data_config = OmegaConf.load('configs/data/wind_data.yaml')
    loaders = factory.create_dataset(data_config)
    
    for loader in loaders:
        sample_batch = next(iter(loader))
        assert 'global_data' in sample_batch
        assert 'query_data' in sample_batch
        assert 'observation_data' in sample_batch
        
        batch_size = data_config.loader.batch_size
        num_chans_global = len(data_config.vars.topo_vars) + len(data_config.vars.nwp_vars)
        
        # check global data shape
        assert sample_batch['global_data']['global_field'].shape[0] == batch_size
        assert sample_batch['global_data']['global_field'].shape[1] == num_chans_global
        assert len(sample_batch['global_data']['global_field'].shape) == 4
        # no nan
        assert not torch.isnan(sample_batch['global_data']['global_field']).any()
        
        # check time_emb
        if data_config.use_time_embedding:
            assert 'time_emb' in sample_batch['global_data']
            assert sample_batch['global_data']['time_emb'].shape[0] == batch_size
            assert len(sample_batch['global_data']['time_emb'].shape) == 2  # default time emb size
            
        # check query shape
        num_queries = sample_batch['query_data']['coords'].shape[1]
        for key, data in sample_batch['query_data'].items():
            if torch.is_tensor(data):
                assert data.shape[0] == batch_size
                assert data.shape[1] == num_queries
                if key == 'coords':
                    assert data.shape[2] == 2
                elif key == 'geo':
                    # TODO: Define what should be in geo
                    pass
                elif key == 'topo':
                    assert data.shape[2] == len(data_config.vars.topo_vars)
                    assert tuple(data.shape[3:]) == tuple(data_config.vars.local_patch_size)
                    
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
                    
                
            