from oat.model_module import CNNTopographyEncoder

import torch

def test_simple_cnn_topographic_encoder():

    in_channels = 8
    out_channels = 256
    topo_enc = CNNTopographyEncoder(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_sizes=5,
                         target_hw=[(24,24), (10,10)],
                         channels=[16,16],
                         act_fn="ReLU",
                         use_batchnorm=True)

    dummy_data = torch.randn(in_channels, 32, 32)

    out = topo_enc(dummy_data)

    assert out.shape[-1] == out_channels


# TODO add tests parametrized by various configs

