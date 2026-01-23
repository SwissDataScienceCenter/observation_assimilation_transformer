from oat.model_module import GatedLinearUnitFusionLayer

import torch
import pytest

@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("residual_learning", [False, True])
def test_glu_fusion_layer(dropout, residual_learning):

    fusion_layer = GatedLinearUnitFusionLayer(in_channels=8,
                                            out_channels=32,
                                            hidden_channels=52,
                                            n_blocks=3,
                                            dropout_p=dropout,
                                            residual_learning=residual_learning)

    sample_input_a = torch.randn(32,12,8)
    sample_input_b = torch.randn(32,12,8)

    out = fusion_layer(a=sample_input_a, b=sample_input_b)

    assert out.shape[-1] == 32
