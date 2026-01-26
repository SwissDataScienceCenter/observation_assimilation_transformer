from oat.model_module import GatedLinearUnitFusionLayer

import torch
import pytest

@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("residual_learning", [False, True])
@pytest.mark.parametrize("layer_norm", ["FiLM", "AdaLN", "LayerNorm"])
def test_glu_fusion_layer(dropout: float, residual_learning: bool, layer_norm: str) -> None:

    cond_dim = 16
    fusion_layer = GatedLinearUnitFusionLayer(in_channels=[8,8],
                                            out_channels=32,
                                            cond_dim=cond_dim,
                                            dropout_p=dropout,
                                            residual_learning=residual_learning,
                                            norm_layer=layer_norm)

    sample_input_a = torch.randn(32,12,8)
    sample_input_b = torch.randn(32,12,8)
    cond = torch.randn(32, 12, cond_dim)

    out = fusion_layer(sample_input_a, sample_input_b, cond=cond)

    assert out.shape[-1] == 32


@pytest.mark.parametrize("num_channels", [[4,4], [3,4,2], [5,5,5,6]])
@pytest.mark.parametrize("layer_norm", ["FiLM", "AdaLN", "LayerNorm"])
def test_glu_fusion_layer_variable_num_inputs(num_channels: list[int], layer_norm: str) -> None:

    cond_dim = 10
    fusion_layer = GatedLinearUnitFusionLayer(in_channels=num_channels,
                                            out_channels=16,
                                            cond_dim=cond_dim,
                                            norm_layer=layer_norm)
    inputs = [torch.randn(32,12,c) for c in num_channels]
    cond = torch.randn(32, 12, cond_dim)
    out = fusion_layer(*inputs, cond=cond)

    assert out.shape[-1] == 16

