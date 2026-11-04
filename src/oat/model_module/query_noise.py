"""Random Fourier feature noise field for query points."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class NoiseFieldDraw:
    """One draw of random Fourier feature parameters.

    Attributes:
        omega: Frequencies, shape [B, M, C, K, D].
        phase: Phases, shape [B, M, C, K].
        amplitude: Amplitudes, shape [B, M, C, K].
    """

    omega: Tensor
    phase: Tensor
    amplitude: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.omega.shape[0])

    @property
    def num_members(self) -> int:
        return int(self.omega.shape[1])

    @property
    def num_channels(self) -> int:
        return int(self.omega.shape[2])

    @property
    def coord_dim(self) -> int:
        return int(self.omega.shape[4])

    def evaluate(self, coords: Tensor) -> Tensor:
        """Evaluate the noise field at the coordinates.

        Args:
            coords: Coordinates, shape [B, N, D].

        Returns:
            Noise values, shape [B, N, M, C].

        Raises:
            ValueError: If the batch size or coordinate dimension of the
                coordinates does not match the draw.
        """
        if coords.shape[0] != self.batch_size or coords.shape[-1] != self.coord_dim:
            raise ValueError(
                f"Coordinates have shape {tuple(coords.shape)}. "
                f"The draw needs batch size {self.batch_size} "
                f"and coordinate dimension {self.coord_dim}."
            )
        num_freq = self.omega.shape[3]
        scale = math.sqrt(2.0 / num_freq)
        members = []
        for m in range(self.num_members):
            arg = torch.einsum("bnd,bckd->bnck", coords, self.omega[:, m])
            arg = arg + self.phase[:, m].unsqueeze(1)
            weighted = self.amplitude[:, m].unsqueeze(1) * torch.cos(arg)
            members.append(scale * weighted.sum(dim=-1))
        return torch.stack(members, dim=2)


class RFFNoiseField(nn.Module):
    """Gaussian-kernel noise field made with random Fourier features.

    Each channel has unit variance and a Gaussian correlation with one length
    scale. Channel c uses the scale `length_scales_km[c // channels_per_scale]`.
    """

    def __init__(
        self,
        length_scales_km: Sequence[float],
        km_per_unit: float,
        channels_per_scale: int,
        num_frequencies: int,
    ) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        self.num_channels = len(length_scales_km) * channels_per_scale
        scales = [
            length / km_per_unit
            for length in length_scales_km
            for _ in range(channels_per_scale)
        ]
        self.register_buffer(
            "length_scales", torch.tensor(scales, dtype=torch.float64), persistent=False
        )

    def draw(
        self,
        batch_size: int,
        num_members: int,
        coord_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> NoiseFieldDraw:
        """Draw the random parameters.

        Returns:
            A draw with omega [B, M, C, K, D], phase and amplitude [B, M, C, K].
        """
        shape = (batch_size, num_members, self.num_channels, self.num_frequencies)
        kwargs = {"device": device, "dtype": dtype, "generator": generator}
        xi = torch.randn(*shape, coord_dim, **kwargs)  # type: ignore[call-overload]
        phase = torch.rand(*shape, **kwargs) * (2 * math.pi)  # type: ignore[call-overload]
        amplitude = torch.randn(*shape, **kwargs)  # type: ignore[call-overload]
        length = self.length_scales.to(device=device, dtype=dtype)
        omega = xi / length.view(1, 1, -1, 1, 1)
        return NoiseFieldDraw(omega=omega, phase=phase, amplitude=amplitude)
