from oat.model_module import GatedLinearUnitFusionLayer

import torch

def test_glu_fusion_layer():

    fusion_layer = GatedLinearUnitFusionLayer(in_channels=8,
                                            out_channels=32,
                                            hidden_channels=52,
                                            n_blocks=3)

    sample_input_a = torch.randn(32,12,8)
    sample_input_b = torch.randn(32,12,8)

    out = fusion_layer(a=sample_input_a, b=sample_input_b)

    assert out.shape[-1] == 32
