import sys
from pathlib import Path

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

from oat.baselines.ootg_likelihood import MixedVariableLikelihood, decoder_out_dim


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
    xy_enc = MLP(
        in_dim=x_embed_dim + dim_y + 1,
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
