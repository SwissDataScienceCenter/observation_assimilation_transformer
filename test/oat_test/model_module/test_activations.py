from oat.model_module.activations import get_activation

import pytest
import torch

def test_invalid_activation():

    with pytest.raises(ValueError):
        get_activation(name='Linear')

def test_activation_with_no_args():

    x = torch.randn(16,2)

    act = get_activation(name="ReLU")

    act(x)

def test_activation_with_args():

    x = torch.randn(16,2)

    act = get_activation(name="LeakyReLU", negative_slope=0.5)

    act(x)
