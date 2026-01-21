import torch.nn as nn

def get_activation(name: str, **kwargs):

    allowed_activations = {
        "ReLU",
        "LeakyReLU",
        "GELU",
        "Sigmoid",
        "Tanh",
        "SiLU",
        "Softplus",
        "PReLU",
    }
    if name not in allowed_activations:
        raise ValueError(f"Activation must be in {allowed_activations}. Given: {name}")
    cls = getattr(nn, name, None)
    if cls is None:
        raise ValueError(f"Activation {name} not found in torch.nn")
    return cls(**kwargs)