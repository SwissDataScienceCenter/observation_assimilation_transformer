from oat.model_module import CNNTopographyEncoder

import pytest
import math
import torch

def test_simple_cnn_topographic_encoder():

    in_channels = 8
    out_channels = 256
    initial_hw = (32,32)
    topo_enc = CNNTopographyEncoder(in_channels=in_channels,
                         out_channels=out_channels,
                         n_layers=3,
                         initial_hw=initial_hw,
                         kernel_sizes=5,
                         act_fn="ReLU",
                         use_batchnorm=True)

    sample_input = torch.randn(1, in_channels, initial_hw[0], initial_hw[1])

    out = topo_enc(sample_input)

    assert out.shape[-1] == out_channels

    for i in range(3):
        # Assert channels are doubling
        conv2d = topo_enc.conv_blocks[i][0]
        assert conv2d.out_channels == in_channels * 2**i
        # Assert H and W are halving
        ada_pool = topo_enc.conv_blocks[i][3]
        assert ada_pool.output_size == \
            (math.ceil(initial_hw[0] / 2**i), math.ceil(initial_hw[1] / 2**i))

# TODO add tests parametrized by various configs


def test_cnn_with_different_dims():
    in_channels = 8
    out_channels = 256
    initial_hw = (32,32)
    topo_enc = CNNTopographyEncoder(in_channels=in_channels,
                         out_channels=out_channels,
                         n_layers=3,
                         initial_hw=initial_hw,
                         kernel_sizes=5,
                         act_fn="ReLU",
                         use_batchnorm=True)

    stations = 10
    batch_size = 16
    # Example with [B,S,C,H,W]
    sample_input = torch.randn(batch_size, stations, 
                               in_channels, initial_hw[0], initial_hw[1])

    out = topo_enc(sample_input)
    assert out.shape[0] == batch_size
    assert out.shape[1] == stations
    assert out.shape[2] == out_channels

    # Example with [S,C,H,W]
    sample_input = torch.randn(stations, in_channels, initial_hw[0], initial_hw[1])

    out = topo_enc(sample_input)
    assert out.shape[0] == stations
    assert out.shape[1] == out_channels


configs = [
        {
            "in_channels": 5,
            "out_channels": 16,
            "n_layers": 2,
            "kernel_sizes": 3,
            "initial_hw": (32, 32)
        },
        {
            "in_channels": 5,
            "out_channels": 16,
            "n_layers": 2,
            "kernel_sizes": 3,
            "target_hw": [(16,16), (10,10)]
        },
        {
            "in_channels": 5,
            "out_channels": 16,
            "n_layers": 2,
            "kernel_sizes": 3,
            "target_hw": [(16,16), (10,10)],
            "use_batchnorm": False
        },
        {
            "in_channels": 5,
            "out_channels": 16,
            "n_layers": 2,
            "kernel_sizes": 3,
            "target_hw": [(16,16), (10,10)],
            "act_fn": "LeakyReLU",
            "act_fn_kwargs": {"negative_slope": 0.05}
        }
    ]
@pytest.mark.parametrize("cfg", configs)
def test_cnn_with_batch_fixture(batch_fixture, cfg):

    sample_input = batch_fixture['query_data']['topo']

    topo_enc = CNNTopographyEncoder(**cfg)

    out = topo_enc(sample_input)

    assert out.shape[-1] == cfg['out_channels']
