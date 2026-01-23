from oat.model_module import transformer
import torch
import pytest
from typing import Tuple, Optional

@pytest.fixture
def setup_data() -> Tuple[int, int, int, int, int, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    torch.manual_seed(0)
    B = 2  # Batch size
    N_kv = 10  # Number of key/value tokens
    N_q = 5   # Number of query tokens
    D = 16    # Model dimension
    num_heads = 4

    Z_kv = torch.randn(B, N_kv, D)
    Z_q = torch.randn(B, N_q, D)

    mask_src = None
    mask_tgt = None
    mask_memory = None

    dist_src = None
    dist_tgt = None
    dist_memory = None

    return (B, N_kv, N_q, D, num_heads, Z_kv, Z_q,
            mask_src, mask_tgt, mask_memory,
            dist_src, dist_tgt, dist_memory)
    
def test_oat_forward(setup_data: Tuple[int, int, int, int, int, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]) -> None:
    (B, N_kv, N_q, D, num_heads, Z_kv, Z_q,
     mask_src, mask_tgt, mask_memory,
     dist_src, dist_tgt, dist_memory) = setup_data

    oat = transformer.ObservationAssimilationTransformer(
        d_model=D,
        nhead=num_heads,
        d_geo_bias=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        dropout=0.,
    )

    oat_out = oat(
        Z_kv,
        Z_q,
        src_mask=mask_src,
        tgt_mask=mask_tgt,
        memory_mask=mask_memory,
        dist_src=dist_src,
        dist_tgt=dist_tgt,
        dist_memory=dist_memory,
    )

    assert oat_out.shape == (B, N_q, D)
    
def test_geometry_aware_encoder_layer(setup_data: Tuple[int, int, int, int, int, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]) -> None:
    (B, N_kv, N_q, D, num_heads, Z_kv, Z_q,
     mask_src, mask_tgt, mask_memory,
     dist_src, dist_tgt, dist_memory) = setup_data

    encoding_layer = transformer.GeometryAwareEncoderLayer(
        d_model=D,
        nhead=num_heads,
        d_geo_bias=8,
        dim_feedforward=64,
        dropout=0.,
    )

    enc_out = encoding_layer(Z_kv, src_mask=mask_src, dist_src=dist_src)
    assert enc_out.shape == (B, N_kv, D)
    
def test_geometry_aware_multihead_attention(setup_data: Tuple[int, int, int, int, int, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]) -> None:
    (B, N_kv, N_q, D, num_heads, Z_kv, Z_q,
     mask_src, mask_tgt, mask_memory,
     dist_src, dist_tgt, dist_memory) = setup_data

    attn_module = transformer.GeometryAwareMHA(
        embed_dim=D,
        num_heads=num_heads,
        d_geo_bias=8,
        dropout=0.,
    )

    Z_q_tilde, _ = attn_module(Z_q, Z_kv, Z_kv, dist_memory, attn_mask=mask_memory)
    attn_module.sparse = True
    Z_q_tilde_sparse, _ = attn_module(Z_q, Z_kv, Z_kv, dist_memory, attn_mask=mask_memory)

    assert Z_q_tilde.shape == (B, N_q, D)
    assert Z_q_tilde_sparse.shape == (B, N_q, D)
    assert torch.allclose(Z_q_tilde, Z_q_tilde_sparse, atol=1e-6)