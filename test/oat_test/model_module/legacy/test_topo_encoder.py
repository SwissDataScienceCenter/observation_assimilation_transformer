import pytest

from oat.model_module import CNNTopographyEncoder


def test_cnn_topography_encoder_is_deprecated():
    with pytest.raises(NotImplementedError, match="deprecated"):
        CNNTopographyEncoder(
            in_channels=8,
            out_channels=256,
            n_layers=3,
            initial_hw=(32, 32),
            kernel_sizes=5,
            act_fn="ReLU",
            use_batchnorm=True,
        )
