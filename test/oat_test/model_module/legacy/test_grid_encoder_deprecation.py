import pytest
import torch

from oat.model_module.global_encoder import InterpolationGlobalEncoder
from oat.model_module.topographic_encoder import CNNTopographyEncoder


def test_interpolation_global_encoder_is_deprecated():
    with pytest.raises(NotImplementedError, match="deprecated"):
        InterpolationGlobalEncoder(
            field_extent={"min": torch.zeros(2), "max": torch.ones(2)}
        )


def test_cnn_topography_encoder_is_deprecated():
    with pytest.raises(NotImplementedError, match="deprecated"):
        CNNTopographyEncoder(in_channels=1, out_channels=8, n_layers=1, kernel_sizes=3)
