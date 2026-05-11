from oat.model_module import GatedLinearUnitFusionLayer, MultiCondGLUFusionLayer

import torch
import pytest

@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("residual_learning", [False, True])
@pytest.mark.parametrize("layer_norm", ["FiLM", "AdaLN", "LayerNorm"])
def test_glu_fusion_layer(dropout: float, residual_learning: bool, layer_norm: str) -> None:

    cond_dim = 16
    fusion_layer = GatedLinearUnitFusionLayer(
        in_channels=[8,8],
        out_channels=32,
        cond_dim=cond_dim,
        dropout_p=dropout,
        residual_learning=residual_learning,
        norm_layer=layer_norm)

    sample_input_a = torch.randn(32, 12, 8)
    sample_input_b = torch.randn(32, 12, 8)
    cond = torch.randn(32, 12, cond_dim)
    out = fusion_layer(sample_input_a, sample_input_b, cond=cond)

    assert out.shape[-1] == 32


@pytest.mark.parametrize("num_channels", [[4, 4], [3, 4, 2], [5, 5, 5, 6]])
@pytest.mark.parametrize("layer_norm", ["FiLM", "AdaLN", "LayerNorm"])
@pytest.mark.parametrize("residual_learning", [False, True])
@pytest.mark.parametrize("num_blocks", [1, 2])
@pytest.mark.parametrize("norm_first", [False, True])
def test_glu_fusion_layer_variable_num_inputs(num_channels: list[int], 
                                              layer_norm: str, 
                                              residual_learning: bool,
                                              num_blocks: int,
                                              norm_first: bool) -> None:

    cond_dim = 10
    fusion_layer = GatedLinearUnitFusionLayer(
        in_channels=num_channels,
        n_blocks=num_blocks,
        out_channels=16,
        cond_dim=cond_dim,
        norm_layer=layer_norm,
        residual_learning=residual_learning,
        norm_first=norm_first)
    inputs = [torch.randn(32, 12, c) for c in num_channels]
    cond = torch.randn(32, 12, cond_dim)
    out = fusion_layer(*inputs, cond=cond)

    assert out.shape[-1] == 16

@pytest.mark.parametrize("num_channels", [[4, 4], [3, 4, 2], [5, 5, 5, 6]])
@pytest.mark.parametrize("layer_norm", ["FiLM", "AdaLN", "LayerNorm"])
@pytest.mark.parametrize("residual_learning", [False, True])
@pytest.mark.parametrize("num_blocks", [1, 2])
@pytest.mark.parametrize("norm_first", [False, True])
@pytest.mark.parametrize("cond_dims", [[10], [10, 15], [10, 15, 20]])
def test_glu_fusion_layer_variable_num_inputs_and_conds(num_channels: list[int], 
                                              layer_norm: str, 
                                              residual_learning: bool,
                                              num_blocks: int,
                                              norm_first: bool,
                                              cond_dims: list[int]) -> None:
    
    fusion_layer = MultiCondGLUFusionLayer(
        in_channels=num_channels,
        n_blocks=num_blocks,
        out_channels=16,
        cond_dims=cond_dims,
        norm_layer=layer_norm,
        residual_learning=residual_learning,
        norm_first=norm_first)
    
    
    inputs = [torch.randn(32, 12, c) for c in num_channels]
    cond = [torch.randn(32, 12, d) for d in cond_dims]
    out = fusion_layer(*inputs, cond=cond)

    assert out.shape[-1] == 16