from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# TODO move in a shared location?
def get_activation(name: str, **kwargs):
    cls = getattr(nn, name, None)
    if cls is None:
        raise ValueError(f"Activation {name} not found in torch.nn")
    return cls(**kwargs)


class TopographyEncoder(nn.Module, ABC):
    def __init__(self, *, embedding_dim: int):
        super().__init__()

        self.embedding_dim = embedding_dim

    def forward(self, topography_patch: torch.Tensor) -> torch.Tensor:
        return self._encode_patch(topography_patch)

    @abstractmethod
    def _encode_patch(self, topography_patch: torch.Tensor) -> torch.Tensor:
        """Encode a topography patch into a feature representation."""
        pass


class CNNTopographyEncoder(TopographyEncoder):

    def __init__(self, *,
                 embedding_dim: int,
                 n_kernels: int,
                 kernel_size: int,
                 act_fn: str):
        super().__init__(embedding_dim=embedding_dim)

        # self.convs = nn.Conv2d()
        # TODO finish