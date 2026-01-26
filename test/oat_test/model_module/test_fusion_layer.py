from oat.model_module import GatedLinearUnitFusionLayer

import torch
import pytest

@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("residual_learning", [False, True])
def test_glu_fusion_layer(dropout, residual_learning):

    fusion_layer = GatedLinearUnitFusionLayer(in_channels=[8,8],
                                            out_channels=32,
                                            hidden_channels=52,
                                            n_blocks=3,
                                            dropout_p=dropout,
                                            residual_learning=residual_learning)

    sample_input_a = torch.randn(32,12,8)
    sample_input_b = torch.randn(32,12,8)

    out = fusion_layer(sample_input_a, sample_input_b)

    assert out.shape[-1] == 32


@pytest.mark.parametrize("num_channels", [[4,4], [3,4,2], [5,5,5,6]])
def test_glu_fusion_layer_variable_num_inputs(num_channels):

    fusion_layer = GatedLinearUnitFusionLayer(in_channels=num_channels,
                                            out_channels=16,
                                            hidden_channels=16,
                                            n_blocks=3)

    inputs = [torch.randn(32,12,c) for c in num_channels]

    out = fusion_layer(*inputs)

    assert out.shape[-1] == 16

