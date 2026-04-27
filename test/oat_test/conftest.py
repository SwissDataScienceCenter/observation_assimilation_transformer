import pytest
from pathlib import Path
import torch
import pickle

from oat.model_module.model import (
    GlobalEncoderConfig,
    TopographicEncoderConfig,
    FusionLayerConfig,
    MaskingConfig,
    TransformerConfig,
    OutputHeadConfig,
    OATConfig,
    UninformedOATConfig
)
from oat.data_module.transform import Transform

@pytest.fixture
def batch_fixture() -> dict[str, dict[str, torch.Tensor]]:
    """Provides a dummy batch based on the saved batch_shapes.pickle table."""
    # return generate_batch_from_table()
    
    batch_fixture_file = Path(__file__).resolve().parent / "example_batch_fixture.pickle"
    with open(batch_fixture_file, "rb") as f:
        batch = pickle.load(f)
    return batch

@pytest.fixture
def model_cfg_fixture(batch_fixture):
    global_enc_cfg = GlobalEncoderConfig(
        in_channels=batch_fixture['global_data']['global_field'].shape[1],
        field_extent_min= [2255500.0, 840500.0],
        field_extent_max= [2964500.0, 1479500.0],
        out_channels=4,
        channels=(4,8)
    )
    
    topography_enc_cfg = TopographicEncoderConfig(
        in_channels=batch_fixture['observation_data']['topo'].shape[2],
        out_channels=4
    )
    
    fusion_layer_cfg = FusionLayerConfig(out_channels=4, hidden_channels=4)
    masking_cfg = MaskingConfig()
    transformer_cfg = TransformerConfig(
        d_geo_bias=batch_fixture['observation_data']['geo'].shape[-1],
        dim_feedforward=4,
        d_model=4,
        dim_geo_bias_feedforward=4
    )
    output_head_cfg = OutputHeadConfig(
        in_features=4,
        variable_to_distribution={f"var_{i}": "normal" for i in range(batch_fixture['observation_data']['obs'].shape[-1])},
        loss_scaling=None,
        crps_mc_samples=50,
        use_explicit_innovation=False # TODO Set to true once implemented
    )
    
    oat_cfg = OATConfig(
        global_encoder=global_enc_cfg,
        topographic_encoder=topography_enc_cfg,
        fusion_layer=fusion_layer_cfg,
        transformer=transformer_cfg,
        masking=masking_cfg,
        output_head=output_head_cfg,
        num_samples=10
    )
    
    return oat_cfg

@pytest.fixture
def uninformed_model_cfg_fixture(batch_fixture):
    global_enc_cfg = GlobalEncoderConfig(
        in_channels=batch_fixture['global_data']['global_field'].shape[1],
        field_extent_min= [2255500.0, 840500.0],
        field_extent_max= [2964500.0, 1479500.0],
        out_channels=4,
        channels=(4,8)
    )
    
    topography_enc_cfg = TopographicEncoderConfig(
        in_channels=batch_fixture['observation_data']['topo'].shape[2],
        out_channels=4
    )
    
    fusion_layer_cfg = FusionLayerConfig(out_channels=4, hidden_channels=4)

    output_head_cfg = OutputHeadConfig(
        in_features=4,
        variable_to_distribution={f"var_{i}": "normal" for i in range(batch_fixture['observation_data']['obs'].shape[-1])},
        loss_scaling=None,
        crps_mc_samples=50,
        use_explicit_innovation=False # TODO Set to true once implemented
    )
    
    uninformed_oat_cfg = UninformedOATConfig(
        global_encoder=global_enc_cfg,
        topographic_encoder=topography_enc_cfg,
        fusion_layer=fusion_layer_cfg,
        output_head=output_head_cfg,
        num_samples=10
    )
    return uninformed_oat_cfg

@pytest.fixture
def transform_fixture() -> tuple[Transform, Transform]:
    """Provides a dummy transform."""
    
    in_pickle_file = Path(__file__).resolve().parent / "input_transform_fixture.pickle"
    out_pickle_file = Path(__file__).resolve().parent / "output_transform_fixture.pickle"
    
    with open(in_pickle_file, "rb") as f:
        in_transform = pickle.load(f) 
        
    with open(out_pickle_file, "rb") as f:
        out_transform = pickle.load(f) 
        
    return in_transform, out_transform
