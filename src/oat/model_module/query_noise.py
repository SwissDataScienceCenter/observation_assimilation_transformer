"""Random Fourier feature noise field for query points."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from oat.data_module.support import _to_bounds
from oat.model_module.activations import get_activation
from oat.model_module.output_head import OutputHead


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


class SampleHead(nn.Module):
    """MLP head that maps a query feature and a noise vector to one sample.

    The model trains this head with the fair ensemble CRPS.
    The input is the concatenation [h ; eps]. The head copies h for each member.
    The output for variable v is z_v, or lb + ST-ReLU(z_v - lb) if the support of
    v has a finite lower bound lb. ST-ReLU gives exact zeros in the forward pass
    and a gradient of 1 in the backward pass.
    """

    def __init__(
        self,
        in_features: int,
        noise_channels: int,
        output_head: OutputHead,
        mlp_hidden_layers: int = 2,
        vs_weight: float = 0.0,
        vs_p: float = 0.5,
        mixing: str = "channels",
        num_scales: int = 1,
        nugget_index: int = 0,
        mixture_init_logit: float = -4.0,
        vs_warmup_epochs: int = 0,
        vs_ramp_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.vs_weight = vs_weight
        self.vs_p = vs_p
        if vs_warmup_epochs < 0 or vs_ramp_epochs < 0:
            raise ValueError("vs_warmup_epochs and vs_ramp_epochs must be 0 or more.")
        self.vs_warmup_epochs = vs_warmup_epochs
        self.vs_ramp_epochs = vs_ramp_epochs
        # Plain float, not a buffer: no state_dict key. The predictor sets it.
        self.epoch = 0.0
        if mixing not in ("channels", "scale_mixture"):
            raise ValueError(
                f"mixing must be 'channels' or 'scale_mixture', not {mixing!r}."
            )
        self.mixing = mixing
        self.num_scales = num_scales
        head_noise_channels = noise_channels
        if mixing == "scale_mixture":
            if num_scales < 1 or noise_channels % num_scales != 0:
                raise ValueError(
                    f"noise_channels ({noise_channels}) must be a multiple of "
                    f"num_scales ({num_scales})."
                )
            self.channels_per_scale = noise_channels // num_scales
            head_noise_channels = self.channels_per_scale
            a0 = torch.full((num_scales,), float(mixture_init_logit))
            a0[nugget_index] = 0.0
            # The logits are a0 + delta. The weight decay pulls delta to 0, thus
            # it pulls pi back to the init (nugget only), not to a uniform pi.
            self.register_buffer("mixture_a0", a0)
            self.mixture_delta = nn.Parameter(torch.zeros(num_scales))
        names = list(output_head.variable_to_distribution)
        self.variable_names = names
        lower = [
            _to_bounds(output_head.distribution_mappers[v].support).lower_bound
            for v in names
        ]
        self.register_buffer("lower_bounds", torch.tensor(lower), persistent=False)
        self.register_buffer(
            "loss_scaling",
            torch.as_tensor(output_head.loss_scaling, dtype=torch.float32).clone(),
            persistent=False,
        )
        # Same loop as MLPTopographyEncoder and PointGlobalEncoder. Keep the copy
        # until a shared build_mlp helper exists.
        layers: list[nn.Module] = []
        width = in_features + head_noise_channels
        for _ in range(mlp_hidden_layers):
            layers += [nn.Linear(width, in_features), get_activation("GELU")]
            width = in_features
        layers.append(nn.Linear(width, len(names)))
        self.mlp = nn.Sequential(*layers)

    def train_vs_weight(self) -> float:
        """VS weight in training for the current epoch.

        0 for epoch < vs_warmup_epochs. Then linear up to vs_weight over
        vs_ramp_epochs. Then vs_weight.
        """
        done = self.epoch - self.vs_warmup_epochs
        if done < 0:
            return 0.0
        if self.vs_ramp_epochs == 0:
            return self.vs_weight
        return self.vs_weight * min(1.0, (done + 1) / self.vs_ramp_epochs)

    @property
    def pi(self) -> Tensor | None:
        """Mixture weight of each length scale, shape [S]. None for `channels`."""
        if self.mixing != "scale_mixture":
            return None
        return torch.softmax(self.mixture_a0 + self.mixture_delta, dim=0)

    def mix_noise(self, eps: Tensor) -> Tensor:
        """Mix the length scales of the noise channels.

        For `scale_mixture`, channel c of the output is
        sum_s sqrt(pi_s) * eps[s, c]. It has unit variance and the correlation
        sum_s pi_s * k_s(d). At one point the output is N(0, I) for all pi, thus
        the CRPS does not change pi in expectation. Only the VS term does.
        The field orders the channels scale first (c // channels_per_scale).

        Args:
            eps: Noise, shape [B, N, M, S*C].

        Returns:
            Noise, shape [B, N, M, C] for `scale_mixture`, else eps.
        """
        pi = self.pi
        if pi is None:
            return eps
        grouped = eps.unflatten(-1, (self.num_scales, self.channels_per_scale))
        weight = pi.sqrt().to(eps.dtype).unsqueeze(-1)  # [S, 1]
        return (grouped * weight).sum(dim=-2)

    def forward(self, h: Tensor, eps: Tensor) -> Tensor:
        """Get samples.

        Args:
            h: Query features, shape [B, N, F].
            eps: Noise, shape [B, N, M, C].

        Returns:
            Samples, shape [B, N, M, V].
        """
        eps = self.mix_noise(eps)
        h_exp = h.unsqueeze(2).expand(-1, -1, eps.shape[2], -1)
        z = self.mlp(torch.cat([h_exp, eps], dim=-1))
        lb = self.lower_bounds.to(z.dtype)
        shifted = z - lb
        st_relu = shifted + (shifted.clamp(min=0.0) - shifted).detach()
        return torch.where(torch.isfinite(lb), lb + st_relu, z)

    def loss(
        self,
        members: Tensor,
        target: Tensor,
        mask: Tensor,
        coords_km: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], Tensor, dict[str, Tensor]]:
        """Get the fair ensemble CRPS plus the optional variogram score.

        The CRPS is normalized for each sample. For each variable v, the loss is
        (L_crps_v + w * L_vs_v) * loss_scaling_v. In training, w is
        train_vs_weight(). In eval mode, w is vs_weight.

        Args:
            members: Samples, shape [B, N, M, V].
            target: Targets, shape [B, N, V]. Masked entries can be NaN.
            mask: Valid entries, shape [B, N, V].
            coords_km: Query coordinates in km, shape [B, N, D]. The model needs
                them only if vs_weight is more than 0.

        Returns:
            Detached loss for each variable, the total loss, and the detached
            variogram score L_vs_v for each variable. The last value is an empty
            dict if vs_weight is 0.
        """
        # Import here: oat.predictor_module imports model.py, which imports this file.
        from oat.predictor_module.functional_metrics import (
            fair_variogram_score,
            mc_crps,
        )

        # Validation and test always use the full weight.
        weight = self.train_vs_weight() if self.training else self.vs_weight
        use_vs = weight > 0
        if use_vs and coords_km is None:
            raise ValueError("coords_km is needed when vs_weight is more than 0.")
        valid = mask > 0.5
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        score = mc_crps(safe_target, members, mask=None, aggregation="none")
        score = torch.where(valid, score, torch.zeros_like(score))
        vs_score = has_pair = None
        if use_vs:
            assert coords_km is not None
            vs_score, has_pair = fair_variogram_score(
                target, members, mask, coords_km, p=self.vs_p
            )
        per_var: dict[str, Tensor] = {}
        vs_per_var: dict[str, Tensor] = {}
        total = torch.zeros((), device=members.device, dtype=members.dtype)
        for i, name in enumerate(self.variable_names):
            loss_v = OutputHead.loss_normalization_per_sample(
                score[..., i : i + 1], mask[..., i : i + 1]
            )
            if vs_score is not None and has_pair is not None:
                # Mean over the samples that have a valid pair. 0 if none.
                n_pair = has_pair[:, i].sum().clamp(min=1).to(vs_score.dtype)
                vs_v = vs_score[:, i].sum() / n_pair
                vs_per_var[name] = vs_v.detach()
                loss_v = loss_v + weight * vs_v
            loss_v = loss_v * self.loss_scaling[i].to(members.dtype)
            per_var[name] = loss_v.detach()
            total = total + loss_v
        return per_var, total, vs_per_var
