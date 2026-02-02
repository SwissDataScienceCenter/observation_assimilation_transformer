from oat.model_module import attention
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
    edges_src = torch.nonzero(~mask_src, as_tuple=False)
    edges_tgt = torch.nonzero(~mask_tgt, as_tuple=False)
    edges_memory = torch.nonzero(~mask_memory, as_tuple=False)
    dist_src = torch.abs(torch.randn(B, N_kv, N_kv, d_geo_bias))
    dist_tgt = torch.abs(torch.randn(B, N_q, N_q, d_geo_bias))
    dist_memory = torch.abs(torch.randn(B, N_q, N_kv, d_geo_bias))
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
        "mask_src": mask_src,
        "mask_tgt": mask_tgt,
        "mask_memory": mask_memory,
        "edges_src": edges_src,
        "edges_tgt": edges_tgt,
        "edges_memory": edges_memory,
    }
    return data


def test_geometry_aware_multihead_attention(setup_data: Dict[str, Any]) -> None:
    B = setup_data["B"]
    N_q = setup_data["N_q"]
    D = setup_data["D"]
    num_heads = setup_data["num_heads"]
    d_geo_bias = setup_data["d_geo_bias"]
    Z_kv = setup_data["Z_kv"]
    Z_q = setup_data["Z_q"]
    g_src = setup_data["g_src"]
    g_tgt = setup_data["g_tgt"]
    mask_memory = setup_data["mask_memory"]
    edges_memory = setup_data["edges_memory"]
    dropout = setup_data["dropout"]

    attn_module = attention.GeometryAwareMHA(
        embed_dim=D,
        num_heads=num_heads,
        d_geo_bias=d_geo_bias,
        dropout=dropout,
    )

    Z_q_tilde, _ = attn_module(Z_q, Z_kv, Z_kv, attn_edge_index=None, geo_k=g_src, geo_q=g_tgt)
    attn_module.sparse = True
    Z_q_tilde_sparse, _ = attn_module(Z_q, Z_kv, Z_kv, attn_edge_index=edges_memory, geo_k=g_src, geo_q=g_tgt)

    assert Z_q_tilde.shape == (B, N_q, D)
    assert Z_q_tilde_sparse.shape == (B, N_q, D)   
    # if dropout == 0.0:
    #     assert torch.allclose(Z_q_tilde, Z_q_tilde_sparse, atol=1e-5)
    
def test_transformer_gradient(setup_data: Dict[str, Any]) -> None:
    N_q = setup_data["N_q"]
    D = setup_data["D"]
    num_heads = setup_data["num_heads"]
    d_geo_bias = setup_data["d_geo_bias"]
    Z_kv = setup_data["Z_kv"].clone().requires_grad_(True)
    Z_q = setup_data["Z_q"].clone().requires_grad_(True)
    mask_memory = setup_data["mask_memory"]
    edges_memory = setup_data["edges_memory"]
    dropout = setup_data["dropout"]
    g_src = setup_data["g_src"]
    g_tgt = setup_data["g_tgt"]

    attn_module = attention.GeometryAwareMHA(
        embed_dim=D,
        num_heads=num_heads,
        d_geo_bias=d_geo_bias,
        dropout=dropout,
    )
    

    Z_q_tilde, _ = attn_module(Z_q, Z_kv, Z_kv, geo_k=g_src, geo_q=g_tgt)
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
    
    Z_q_tilde_sparse, _ = attn_module(Z_q, Z_kv, Z_kv, attn_edge_index=edges_memory, geo_k=g_src, geo_q=g_tgt)
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