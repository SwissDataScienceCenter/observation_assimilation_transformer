import sys
from pathlib import Path
from typing import Any

import torch

_TNPDIR = Path(__file__).parents[3] / "deps" / "gridded-tnp"
if _TNPDIR.exists() and str(_TNPDIR) not in sys.path:
    sys.path.insert(0, str(_TNPDIR))

from tnp.models.gridded_tnp import OOTGGriddedTNP, OOTGGriddedTNPEncoder
from tnp.models.tnp import TNPDecoder
from tnp.networks.attention_layers import (
    MultiHeadCrossAttentionLayer,
    MultiHeadSelfAttentionLayer,
)
from tnp.networks.embeddings import SphericalHarmonicsEmbedding
from tnp.networks.grid_decoders import MHCAGridDecoder
from tnp.networks.grid_encoders import OOTGPseudoTokenGridEncoder
from tnp.networks.mlp import MLP, MLPWithEmbedding
from tnp.networks.swin_attention import SWINAttentionLayer
from tnp.networks.transformer import GriddedTransformerEncoder
from tnp.utils.helpers import preprocess_observations

from oat.baselines.ootg_likelihood import MixedVariableLikelihood, decoder_out_dim
from oat.model_module.topographic_encoder import CNNTopographyEncoder


class TopoOOTGGriddedTNPEncoder(OOTGGriddedTNPEncoder):
    """OOTG encoder that injects high-res topo into context/target tokens.

    Re-implements the parent forward (which is not cleanly hookable) to
    concatenate topo_encoder(topo_c/topo_t) into the zc/zt inputs before
    xy_encoder. All other behavior is identical to the parent.
    """

    def __init__(self, *args: Any, topo_encoder: CNNTopographyEncoder, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.topo_encoder = topo_encoder

    def _encode_topo(self, topo: torch.Tensor) -> torch.Tensor:
        # CNNTopographyEncoder.forward reshapes with an inferred (-1) dim,
        # which torch refuses when the token count is 0 (ambiguous size).
        # Short-circuit to an empty tensor of the right shape in that case.
        if topo.shape[1] == 0:
            return topo.new_zeros(*topo.shape[:2], self.topo_encoder.out_channels)
        return self.topo_encoder(topo)

    def forward(self, xc: torch.Tensor, yc: torch.Tensor, xc_grid: torch.Tensor, yc_grid: torch.Tensor, xt: torch.Tensor, topo_c: torch.Tensor, topo_t: torch.Tensor) -> torch.Tensor:
        yc, yt = preprocess_observations(xt, yc)

        x = torch.cat((xc, xt), dim=1)
        x_encoded = self.x_encoder(x)
        xc_encoded, xt_encoded = x_encoded.split((xc.shape[1], xt.shape[1]), dim=1)

        y = torch.cat((yc, yt), dim=1)
        y_encoded = self.y_encoder(y)
        yc_encoded, yt_encoded = y_encoded.split((yc.shape[1], yt.shape[1]), dim=1)

        xc_grid_encoded = self.x_encoder(xc_grid)
        yc_grid_encoded = self.y_grid_encoder(yc_grid)

        # High-res topo -> per-token vector, concatenated symmetrically.
        topo_c_encoded = self._encode_topo(topo_c)
        topo_t_encoded = self._encode_topo(topo_t)

        zc = torch.cat((xc_encoded, yc_encoded, topo_c_encoded), dim=-1)
        zt = torch.cat((xt_encoded, yt_encoded, topo_t_encoded), dim=-1)
        zc = self.xy_encoder(zc)
        zt = self.xy_encoder(zt)

        zc_grid = torch.cat((xc_grid_encoded, yc_grid_encoded), dim=-1)
        zc_grid = self.xy_grid_encoder(zc_grid)

        xc_grid, zc_grid = self.grid_encoder(xc, zc, xc_grid, zc_grid)
        zt = self.transformer_encoder(xc_grid, zc_grid, xt, zt)
        return zt


class TopoOOTGGriddedTNP(OOTGGriddedTNP):
    """OOTGGriddedTNP whose forward threads topo_c/topo_t to the encoder."""

    def forward(self, xc: torch.Tensor, yc: torch.Tensor, xc_grid: torch.Tensor, yc_grid: torch.Tensor, xt: torch.Tensor, topo_c: torch.Tensor, topo_t: torch.Tensor) -> torch.Tensor:
        return self.likelihood(
            self.decoder(self.encoder(xc, yc, xc_grid, yc_grid, xt, topo_c, topo_t), xt)
        )


def build_ootg_model(
    dim_y: int = 1,
    dim_gridded_y: int = 2,
    obs_variable_to_distribution: dict[str, str] | None = None,
    embed_dim: int = 128,
    x_embed_dim: int = 128,
    num_heads: int = 8,
    head_dim: int = 16,
    num_layers: int = 5,
    num_legendre_polys: int = 10,
    grid_shape: tuple[int, int] = (32, 32),
    window_sizes: tuple[int, int] = (4, 4),
    shift_sizes: tuple[int, int] = (2, 2),
    top_k_ctot: int = 9,
    use_topo: bool = False,
    topo_in_channels: int | None = None,
    topo_out_channels: int = 32,
    topo_n_layers: int = 3,
    topo_kernel_size: int = 3,
    topo_initial_hw: tuple[int, int] | None = None,
) -> OOTGGriddedTNP:
    """Build OOTGGriddedTNP with per-variable output distributions.

    obs_variable_to_distribution must contain exactly dim_y entries in the same
    order as obs_vars. If None, defaults to all-Normal with 2*dim_y decoder output.

    grid_shape (32, 32) is the pseudo-token grid. window_sizes must divide
    grid_shape evenly; (4, 4) gives 8x8 windows on (32, 32).
    """
    if obs_variable_to_distribution is None:
        obs_variable_to_distribution = {f"var_{i}": "normal" for i in range(dim_y)}

    assert len(obs_variable_to_distribution) == dim_y, (
        f"obs_variable_to_distribution has {len(obs_variable_to_distribution)} entries "
        f"but dim_y={dim_y}"
    )

    out_dim = decoder_out_dim(obs_variable_to_distribution)
    likelihood = MixedVariableLikelihood(obs_variable_to_distribution)

    mhsa = MultiHeadSelfAttentionLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        feedforward_dim=embed_dim,
        norm_first=True,
    )
    mhca = MultiHeadCrossAttentionLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        feedforward_dim=embed_dim,
        norm_first=True,
    )
    swin = SWINAttentionLayer(
        mhsa_layer=mhsa,
        window_sizes=list(window_sizes),
        shift_sizes=list(shift_sizes),
        roll_dims=[-1],
    )
    transformer_enc = GriddedTransformerEncoder(
        num_layers=num_layers,
        grid_decoder=MHCAGridDecoder(mhca_layer=mhca, top_k_ctot=top_k_ctot),
        mhsa_layer=swin,
    )
    grid_enc = OOTGPseudoTokenGridEncoder(
        embed_dim=embed_dim,
        mhca_layer=mhca,
        grid_shape=grid_shape,
    )
    x_enc = MLPWithEmbedding(
        embeddings=[
            SphericalHarmonicsEmbedding(
                active_dims=[-2, -1],
                num_legendre_polys=num_legendre_polys,
                lonlat_dims=[-1, -2],
            )
        ],
        in_dim=num_legendre_polys**2,
        out_dim=x_embed_dim,
        num_layers=0,
        width=x_embed_dim,
    )
    # preprocess_observations appends a mask bit → xy_encoder.in_dim = x_embed_dim + dim_y + 1
    topo_extra = topo_out_channels if use_topo else 0
    xy_enc = MLP(
        in_dim=x_embed_dim + dim_y + 1 + topo_extra,
        out_dim=embed_dim,
        num_layers=2,
        width=embed_dim,
    )
    xy_grid_enc = MLP(
        in_dim=x_embed_dim + dim_gridded_y,
        out_dim=embed_dim,
        num_layers=2,
        width=embed_dim,
    )

    if use_topo:
        assert (
            topo_in_channels is not None and topo_initial_hw is not None
        ), "use_topo=True requires topo_in_channels and topo_initial_hw"
        topo_encoder = CNNTopographyEncoder(
            in_channels=topo_in_channels,
            out_channels=topo_out_channels,
            n_layers=topo_n_layers,
            kernel_sizes=topo_kernel_size,
            initial_hw=topo_initial_hw,
            use_batchnorm=True,
        )
        encoder = TopoOOTGGriddedTNPEncoder(
            transformer_encoder=transformer_enc,
            grid_encoder=grid_enc,
            xy_encoder=xy_enc,
            xy_grid_encoder=xy_grid_enc,
            x_encoder=x_enc,
            topo_encoder=topo_encoder,
        )
        decoder = TNPDecoder(
            z_decoder=MLP(
                in_dim=embed_dim, out_dim=out_dim, num_layers=2, width=embed_dim
            )
        )
        return TopoOOTGGriddedTNP(
            encoder=encoder, decoder=decoder, likelihood=likelihood
        )

    encoder = OOTGGriddedTNPEncoder(
        transformer_encoder=transformer_enc,
        grid_encoder=grid_enc,
        xy_encoder=xy_enc,
        xy_grid_encoder=xy_grid_enc,
        x_encoder=x_enc,
    )
    decoder = TNPDecoder(
        z_decoder=MLP(
            in_dim=embed_dim, out_dim=out_dim, num_layers=2, width=embed_dim
        )
    )
    return OOTGGriddedTNP(encoder=encoder, decoder=decoder, likelihood=likelihood)
