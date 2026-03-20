from oat.model_module import transformer
import torch
import pytest
from typing import Dict, Any

@pytest.fixture(params=[(0.0, True), (0.0, False), (0.1, False),])
def setup_data(request: pytest.FixtureRequest) -> Dict[str, Any]:
    dropout, norm_first = request.param
    torch.manual_seed(0)
    B = 2  # Batch size
    N_kv = 10  # Number of key/value tokens
    N_q = 5   # Number of query tokens
    D = 16    # Model dimension
    num_heads = 4
    d_geo_bias = 8

    Z_kv = torch.randn(B, N_kv, D)
    Z_q = torch.randn(B, N_q, D)

    mask_src = torch.rand(B, N_kv, N_kv) < 0.2
    mask_tgt = torch.rand(B, N_q, N_q) < 0.2
    mask_memory = torch.rand(B, N_q, N_kv) < 0.6  # 60% density
    edges_list_src, edges_list_tgt, edges_list_memory = [], [], []
    for b in range(B):
        # For memory edges
        q_flat_mem, k_flat_mem = torch.nonzero(~mask_memory[b], as_tuple=True)
        edges_list_memory.append(torch.stack([q_flat_mem + b * N_q, k_flat_mem + b * N_kv], dim=0))
        # For target and source self-attention edges
        q_flat_tgt, k_flat_tgt = torch.nonzero(~mask_tgt[b], as_tuple=True)
        edges_list_tgt.append(torch.stack([q_flat_tgt + b * N_q, k_flat_tgt + b * N_q], dim=0))
        q_flat_src, k_flat_src = torch.nonzero(~mask_src[b], as_tuple=True)
        edges_list_src.append(torch.stack([k_flat_src + b * N_kv, q_flat_src + b * N_kv], dim=0))
    edges_src = torch.cat(edges_list_src, dim=1)  # [2, E]
    edges_tgt = torch.cat(edges_list_tgt, dim=1)  # [2, E]
    edges_memory = torch.cat(edges_list_memory, dim=1)  # [2, E]
    
    g_src = torch.randn(B, N_kv, d_geo_bias)
    g_tgt = torch.randn(B, N_q, d_geo_bias)

    data = {
        "B": B,
        "N_kv": N_kv,
        "N_q": N_q,
        "D": D,
        "num_heads": num_heads,
        "d_geo_bias": d_geo_bias,
        "dropout": dropout,
        "norm_first": norm_first,
        "Z_kv": Z_kv,
        "Z_q": Z_q,
        "g_src": g_src,
        "g_tgt": g_tgt,
        "edges_src": edges_src,
        "edges_tgt": edges_tgt,
        "edges_memory": edges_memory,
    }
    return data


def test_oat_sparse(setup_data: Dict[str, Any]) -> None:
    B = setup_data["B"]
    N_q = setup_data["N_q"]
    D = setup_data["D"]
    num_heads = setup_data["num_heads"]
    d_geo_bias = setup_data["d_geo_bias"]
    Z_kv = setup_data["Z_kv"]
    Z_q = setup_data["Z_q"]
    edges_src = setup_data["edges_src"]
    edges_tgt = setup_data["edges_tgt"]
    edges_memory = setup_data["edges_memory"]
    g_src = setup_data["g_src"]
    g_tgt = setup_data["g_tgt"]
    dropout = setup_data["dropout"]
    norm_first = setup_data["norm_first"]

    oat = transformer.ObservationAssimilationTransformer(
        d_model=D,
        nhead=num_heads,
        d_geo_bias=d_geo_bias,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        dropout=dropout,
        norm_first=norm_first,
    )
    oat_sparse = transformer.ObservationAssimilationTransformer(
        d_model=D,
        nhead=num_heads,
        d_geo_bias=d_geo_bias,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        dropout=dropout,
        norm_first=norm_first,
        sparse_attn=True,
    )
    
    dense_params = oat.state_dict()
    oat_sparse.load_state_dict(dense_params)

    oat_out = oat(
        Z_kv,
        Z_q,
        src_mask_edges=edges_src,
        tgt_mask_edges=edges_tgt,
        memory_mask_edges=edges_memory,
        g_src=g_src,
        g_tgt=g_tgt,
    )
    
    oat_sparse_out = oat_sparse(
        Z_kv,
        Z_q,
        src_mask_edges=edges_src,
        tgt_mask_edges=edges_tgt,
        memory_mask_edges=edges_memory,
        g_src=g_src,
        g_tgt=g_tgt,
    )

    assert oat_out.shape == (B, N_q, D)
    assert oat_sparse_out.shape == (B, N_q, D)
    
    if dropout == 0.0:
        assert torch.allclose(oat_out, oat_sparse_out, atol=1e-5)
    
    
    
    
def test_oat_forward(setup_data: Dict[str, Any]) -> None:
    B = setup_data["B"]
    N_q = setup_data["N_q"]
    D = setup_data["D"]
    num_heads = setup_data["num_heads"]
    d_geo_bias = setup_data["d_geo_bias"]
    Z_kv = setup_data["Z_kv"]
    Z_q = setup_data["Z_q"]
    edges_src = setup_data["edges_src"]
    edges_tgt = setup_data["edges_tgt"]
    edges_memory = setup_data["edges_memory"]
    g_src = setup_data["g_src"]
    g_tgt = setup_data["g_tgt"]
    dropout = setup_data["dropout"]
    norm_first = setup_data["norm_first"]

    oat = transformer.ObservationAssimilationTransformer(
        d_model=D,
        nhead=num_heads,
        d_geo_bias=d_geo_bias,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        dropout=dropout,
        norm_first=norm_first,
    )

    oat_out = oat(
        Z_kv,
        Z_q,
        src_mask_edges=edges_src,
        tgt_mask_edges=edges_tgt,
        memory_mask_edges=edges_memory,
        g_src=g_src,
        g_tgt=g_tgt,
    )

    assert oat_out.shape == (B, N_q, D)

    
def test_geometry_aware_encoder_layer(setup_data: Dict[str, Any]) -> None:
    B = setup_data["B"]
    N_kv = setup_data["N_kv"]
    D = setup_data["D"]
    num_heads = setup_data["num_heads"]
    d_geo_bias = setup_data["d_geo_bias"]
    Z_kv = setup_data["Z_kv"]
    edges_src = setup_data["edges_src"]
    g_src = setup_data["g_src"]
    dropout = setup_data["dropout"]

    encoding_layer = transformer.GeometryAwareEncoderLayer(
        d_model=D,
        nhead=num_heads,
        d_geo_bias=d_geo_bias,
        dim_feedforward=64,
        dropout=dropout,
    )

    enc_out = encoding_layer(Z_kv, src_mask_edges=edges_src, g_src=g_src)
    assert enc_out.shape == (B, N_kv, D)
    
def test_geometry_aware_decoder_layer(setup_data: Dict[str, Any]) -> None:
    B = setup_data["B"]
    N_q = setup_data["N_q"]
    D = setup_data["D"]
    num_heads = setup_data["num_heads"]
    d_geo_bias = setup_data["d_geo_bias"]
    Z_kv = setup_data["Z_kv"]
    Z_q = setup_data["Z_q"]
    edges_tgt = setup_data["edges_tgt"]
    edges_memory = setup_data["edges_memory"]
    g_tgt = setup_data["g_tgt"]
    g_src = setup_data["g_src"]
    dropout = setup_data["dropout"]

    decoding_layer = transformer.GeometryAwareDecoderLayer(
        d_model=D,
        nhead=num_heads,
        d_geo_bias=d_geo_bias,
        dim_feedforward=64,
        dropout=dropout,
    )

    dec_out = decoding_layer(
        Z_q,
        Z_kv,
        tgt_mask_edges=edges_tgt,
        memory_mask_edges=edges_memory,
        g_tgt=g_tgt,
        g_src=g_src,
    )
    assert dec_out.shape == (B, N_q, D)
    
    
def test_decoder_optional_self_attention(model_cfg_fixture):
    
    
    tr_with_sa = transformer.ObservationAssimilationTransformer(decoder_self_attention=True, num_decoder_layers=1)
    tr_without_sa = transformer.ObservationAssimilationTransformer(decoder_self_attention=False, num_decoder_layers=1)
    
    assert any("self_attn" in name for name, _ in tr_with_sa.decoder.layers.named_modules())
    
    assert not any("self_attn" in name for name, _ in tr_without_sa.decoder.layers.named_modules())