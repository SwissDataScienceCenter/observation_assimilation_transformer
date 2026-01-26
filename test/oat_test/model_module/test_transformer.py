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

    dist_src = torch.abs(torch.randn(B, N_kv, N_kv, d_geo_bias))
    dist_tgt = torch.abs(torch.randn(B, N_q, N_q, d_geo_bias))
    dist_memory = torch.abs(torch.randn(B, N_q, N_kv, d_geo_bias))

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
        "mask_src": mask_src,
        "mask_tgt": mask_tgt,
        "mask_memory": mask_memory,
        "dist_src": dist_src,
        "dist_tgt": dist_tgt,
        "dist_memory": dist_memory,
    }
    return data
    
def test_oat_forward(setup_data: Dict[str, Any]) -> None:
    B = setup_data["B"]
    N_q = setup_data["N_q"]
    D = setup_data["D"]
    num_heads = setup_data["num_heads"]
    d_geo_bias = setup_data["d_geo_bias"]
    Z_kv = setup_data["Z_kv"]
    Z_q = setup_data["Z_q"]
    mask_src = setup_data["mask_src"]
    mask_tgt = setup_data["mask_tgt"]
    mask_memory = setup_data["mask_memory"]
    dist_src = setup_data["dist_src"]
    dist_tgt = setup_data["dist_tgt"]
    dist_memory = setup_data["dist_memory"]
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
        src_mask=mask_src,
        tgt_mask=mask_tgt,
        memory_mask=mask_memory,
        dist_src=dist_src,
        dist_tgt=dist_tgt,
        dist_memory=dist_memory,
    )

    assert oat_out.shape == (B, N_q, D)
    
def test_geometry_aware_encoder_layer(setup_data: Dict[str, Any]) -> None:
    B = setup_data["B"]
    N_kv = setup_data["N_kv"]
    D = setup_data["D"]
    num_heads = setup_data["num_heads"]
    d_geo_bias = setup_data["d_geo_bias"]
    Z_kv = setup_data["Z_kv"]
    mask_src = setup_data["mask_src"]
    dist_src = setup_data["dist_src"]
    dropout = setup_data["dropout"]

    encoding_layer = transformer.GeometryAwareEncoderLayer(
        d_model=D,
        nhead=num_heads,
        d_geo_bias=d_geo_bias,
        dim_feedforward=64,
        dropout=dropout,
    )

    enc_out = encoding_layer(Z_kv, src_mask=mask_src, dist_src=dist_src)
    assert enc_out.shape == (B, N_kv, D)
    
    
def test_geometry_aware_decoder_layer(setup_data: Dict[str, Any]) -> None:
    B = setup_data["B"]
    N_q = setup_data["N_q"]
    D = setup_data["D"]
    num_heads = setup_data["num_heads"]
    d_geo_bias = setup_data["d_geo_bias"]
    Z_kv = setup_data["Z_kv"]
    Z_q = setup_data["Z_q"]
    mask_tgt = setup_data["mask_tgt"]
    mask_memory = setup_data["mask_memory"]
    dist_tgt = setup_data["dist_tgt"]
    dist_memory = setup_data["dist_memory"]
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
        tgt_mask=mask_tgt,
        memory_mask=mask_memory,
        dist_tgt=dist_tgt,
        dist_memory=dist_memory,
    )
    assert dec_out.shape == (B, N_q, D)
    
def test_geometry_aware_multihead_attention(setup_data: Dict[str, Any]) -> None:
    B = setup_data["B"]
    N_q = setup_data["N_q"]
    D = setup_data["D"]
    num_heads = setup_data["num_heads"]
    d_geo_bias = setup_data["d_geo_bias"]
    Z_kv = setup_data["Z_kv"]
    Z_q = setup_data["Z_q"]
    mask_memory = setup_data["mask_memory"]
    dist_memory = setup_data["dist_memory"]
    dropout = setup_data["dropout"]

    attn_module = transformer.GeometryAwareMHA(
        embed_dim=D,
        num_heads=num_heads,
        d_geo_bias=d_geo_bias,
        dropout=dropout,
    )

    Z_q_tilde, _ = attn_module(Z_q, Z_kv, Z_kv, dist_memory, attn_mask=mask_memory)
    attn_module.sparse = True
    Z_q_tilde_sparse, _ = attn_module(Z_q, Z_kv, Z_kv, dist_memory, attn_mask=mask_memory)

    assert Z_q_tilde.shape == (B, N_q, D)
    assert Z_q_tilde_sparse.shape == (B, N_q, D)
    if dropout == 0.0:
        assert torch.allclose(Z_q_tilde, Z_q_tilde_sparse, atol=1e-6)
    
def test_transformer_gradient(setup_data: Dict[str, Any]) -> None:
    N_q = setup_data["N_q"]
    D = setup_data["D"]
    num_heads = setup_data["num_heads"]
    d_geo_bias = setup_data["d_geo_bias"]
    Z_kv = setup_data["Z_kv"].clone().requires_grad_(True)
    Z_q = setup_data["Z_q"].clone().requires_grad_(True)
    mask_memory = setup_data["mask_memory"]
    dist_memory = setup_data["dist_memory"]
    dropout = setup_data["dropout"]

    attn_module = transformer.GeometryAwareMHA(
        embed_dim=D,
        num_heads=num_heads,
        d_geo_bias=d_geo_bias,
        dropout=dropout,
    )
    

    Z_q_tilde, _ = attn_module(Z_q, Z_kv, Z_kv, dist_memory, attn_mask=mask_memory)
    loss = Z_q_tilde.sum()
    loss.backward()

    assert Z_q.grad is not None
    assert Z_kv.grad is not None
    assert Z_q.grad.shape == Z_q.shape
    assert Z_kv.grad.shape == Z_kv.shape
    
    # gradients of parameters
    for param in attn_module.parameters():
        assert param.grad is not None
        assert param.grad.shape == param.shape
        assert not torch.all(param.grad == 0)
        
    # reset gradients
    Z_q.grad.zero_()
    Z_kv.grad.zero_()
    for param in attn_module.parameters():
        param.grad.zero_()
    # sparse attention
    attn_module.sparse = True
    
    
    Z_q_tilde_sparse, _ = attn_module(Z_q, Z_kv, Z_kv, dist_memory, attn_mask=mask_memory)
    loss_sparse = Z_q_tilde_sparse.sum()
    loss_sparse.backward()
    
    assert Z_q.grad is not None
    assert Z_kv.grad is not None
    assert Z_q.grad.shape == Z_q.shape
    assert Z_kv.grad.shape == Z_kv.shape
    
    # gradients of parameters
    for param in attn_module.parameters():
        assert param.grad is not None
        assert param.grad.shape == param.shape
        assert not torch.all(param.grad == 0)