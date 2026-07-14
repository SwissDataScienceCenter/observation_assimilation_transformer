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
    edges_memory = setup_data["edges_memory"]
    dropout = setup_data["dropout"]

    attn_module = attention.GeometryAwareMHA(
        embed_dim=D,
        num_heads=num_heads,
        d_geo_bias=d_geo_bias,
        dropout=dropout,
    )

    Z_q_tilde, _ = attn_module(Z_q, Z_kv, Z_kv, attn_edges=edges_memory, geo_k=g_src, geo_q=g_tgt)
    attn_module.sparse_attn = True
    Z_q_tilde_sparse, _ = attn_module(Z_q, Z_kv, Z_kv, attn_edges=edges_memory, geo_k=g_src, geo_q=g_tgt)

    assert Z_q_tilde.shape == (B, N_q, D)
    assert Z_q_tilde_sparse.shape == (B, N_q, D)   
    if dropout == 0.0:
        assert torch.allclose(Z_q_tilde, Z_q_tilde_sparse, atol=1e-5)
    
def test_transformer_gradient(setup_data: Dict[str, Any]) -> None:
    N_q = setup_data["N_q"]
    D = setup_data["D"]
    num_heads = setup_data["num_heads"]
    d_geo_bias = setup_data["d_geo_bias"]
    Z_kv = setup_data["Z_kv"].clone().requires_grad_(True)
    Z_q = setup_data["Z_q"].clone().requires_grad_(True)
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
    

    Z_q_tilde, _ = attn_module(Z_q, Z_kv, Z_kv, attn_edges=edges_memory, geo_k=g_src, geo_q=g_tgt)
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
    attn_module.sparse_attn = True
    
    Z_q_tilde_sparse, _ = attn_module(Z_q, Z_kv, Z_kv, attn_edges=edges_memory, geo_k=g_src, geo_q=g_tgt)
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


def test_geo_attn_mode_construction() -> None:
    add = attention.GeometryAwareMHA(
        embed_dim=16, num_heads=4, d_geo_bias=8, geo_attn_mode="additive"
    )
    assert add.geo_attn_mode == "additive"
    assert add.f_geo is not None
    assert add.geo_kernel is None

    te = attention.GeometryAwareMHA(
        embed_dim=16, num_heads=4, d_geo_bias=8, geo_attn_mode="tetnp"
    )
    assert te.geo_attn_mode == "tetnp"
    assert te.f_geo is None
    assert te.geo_kernel is not None
    # kernel maps (d_geo_bias + num_heads) -> num_heads
    first = te.geo_kernel[0]
    last = te.geo_kernel[-1]
    assert first.in_features == 8 + 4
    assert last.out_features == 4

    van = attention.GeometryAwareMHA(
        embed_dim=16, num_heads=4, d_geo_bias=8, geo_attn_mode="vanilla"
    )
    assert van.geo_attn_mode == "vanilla"
    assert van.f_geo is None
    assert van.geo_kernel is None


def test_geo_attn_mode_validation() -> None:
    with pytest.raises(ValueError):
        attention.GeometryAwareMHA(
            embed_dim=16, num_heads=4, d_geo_bias=8, geo_attn_mode="bogus"
        )
    # tetnp with no geometry has nothing to consume -> error
    with pytest.raises(ValueError):
        attention.GeometryAwareMHA(
            embed_dim=16, num_heads=4, d_geo_bias=None, geo_attn_mode="tetnp"
        )
    # vanilla needs no geometry -> must NOT raise
    attention.GeometryAwareMHA(
        embed_dim=16, num_heads=4, d_geo_bias=None, geo_attn_mode="vanilla"
    )


def test_vanilla_ignores_geometry(setup_data: Dict[str, Any]) -> None:
    d = setup_data
    if d["dropout"] != 0.0:
        pytest.skip("determinism check only needed once")
    attn = attention.GeometryAwareMHA(
        embed_dim=d["D"], num_heads=d["num_heads"],
        d_geo_bias=d["d_geo_bias"], dropout=0.0, geo_attn_mode="vanilla",
    )
    for sparse in (False, True):
        attn.sparse_attn = sparse
        out_a, _ = attn(d["Z_q"], d["Z_kv"], d["Z_kv"],
                        attn_edges=d["edges_memory"], geo_k=d["g_src"], geo_q=d["g_tgt"])
        # different geometry, identical everything else -> identical output
        out_b, _ = attn(d["Z_q"], d["Z_kv"], d["Z_kv"],
                        attn_edges=d["edges_memory"],
                        geo_k=d["g_src"] + 3.0, geo_q=d["g_tgt"] - 1.0)
        assert torch.isfinite(out_a).all()
        assert torch.allclose(out_a, out_b, atol=1e-6)


def test_tetnp_dense_forward(setup_data: Dict[str, Any]) -> None:
    d = setup_data
    if d["dropout"] != 0.0:
        pytest.skip("determinism check only needed once")
    attn = attention.GeometryAwareMHA(
        embed_dim=d["D"], num_heads=d["num_heads"],
        d_geo_bias=d["d_geo_bias"], dropout=0.0, geo_attn_mode="tetnp",
    )
    out, _ = attn(d["Z_q"], d["Z_kv"], d["Z_kv"],
                  attn_edges=d["edges_memory"], geo_k=d["g_src"], geo_q=d["g_tgt"])
    assert out.shape == (d["B"], d["N_q"], d["D"])
    assert torch.isfinite(out).all()
    # the kernel must actually participate in the score (not be inert)
    out.sum().backward()
    assert attn.geo_kernel[0].weight.grad is not None
    assert not torch.all(attn.geo_kernel[0].weight.grad == 0)


def test_tetnp_sparse_matches_dense(setup_data: Dict[str, Any]) -> None:
    d = setup_data
    if d["dropout"] != 0.0:
        pytest.skip("parity requires dropout=0")
    attn = attention.GeometryAwareMHA(
        embed_dim=d["D"], num_heads=d["num_heads"],
        d_geo_bias=d["d_geo_bias"], dropout=0.0, geo_attn_mode="tetnp",
    )
    attn.sparse_attn = False
    dense, _ = attn(d["Z_q"], d["Z_kv"], d["Z_kv"],
                    attn_edges=d["edges_memory"], geo_k=d["g_src"], geo_q=d["g_tgt"])
    attn.sparse_attn = True
    sparse, _ = attn(d["Z_q"], d["Z_kv"], d["Z_kv"],
                     attn_edges=d["edges_memory"], geo_k=d["g_src"], geo_q=d["g_tgt"])
    assert sparse.shape == (d["B"], d["N_q"], d["D"])
    assert torch.allclose(dense, sparse, atol=1e-5)