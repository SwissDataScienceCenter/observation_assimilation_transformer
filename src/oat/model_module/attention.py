import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from oat.model_module.activations import get_activation

class GeometryAwareMHA(nn.Module):
    
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        d_geo_bias: Optional[int] = None,
        geo_bias_hidden_dim: int = 64,
        sparse: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Geometry-aware Multi-Head Attention for neural downscaling.
        
        Implements equations (15)-(25) from the paper.
        
        Automatically handles inputs with arbitrary leading dimensions:
        - Single instance: (N_q, D)
        - Batched: (B, N_q, D)
        - Multi-dimensional: (B1, B2, ..., N_q, D)
        
        Args:
            d_model: Dimension of query embeddings (D_emb)
            d_kv: Dimension of station tokens (D_kv)
            num_heads: Number of attention heads (H)
            d_geo_bias: Dimension of geometric features for bias (D_r)
            dropout: Dropout probability
            geo_bias_hidden_dim: Hidden dimension for geometry bias MLP (default: 64)
            ffn_expansion: Expansion ratio for FFN hidden layer (default: 4, standard in transformers)
        """
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )
        
        self.d_geo_bias = d_geo_bias
        self.sparse = sparse
        # Attention projections (equations 17-19)
        self.W_Q = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.W_K = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.W_V = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        
        # Geometry bias network (equation 15)
        # Maps geometric features to per-head attention biases
        if d_geo_bias:
            self.f_geo = nn.Sequential(
                nn.Linear(d_geo_bias, num_heads * geo_bias_hidden_dim),
                nn.ReLU(),
                nn.Linear(num_heads * geo_bias_hidden_dim, num_heads)
            )
        else:
            self.f_geo = None
        
        self.dropout = nn.Dropout(dropout)
        # Output projection (equation 24)
        self.W_O = nn.Linear(embed_dim, embed_dim)
        
        # Feed-forward network (equation 25)
        # Standard transformer FFN with configurable expansion ratio
        
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dist_qk: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of geometry-aware multi-head attention.
        
        Automatically handles arbitrary leading dimensions.
        
        Args:
            query: Query embeddings, shape (..., N_q, D_emb)
            key: Key embeddings, shape (..., N_kv, D_kv)
            value: Value embeddings, shape (..., N_kv, D_kv)
            dist_kv: Relative geometric features, shape (..., N_q, N_kv, D_r)
            attn_mask: Attention mask, shape (..., N_q, N_kv), optional
                       True/1 = keep, False/0 = mask out
        Returns:
            attn_output: Updated query embeddings, shape (..., N_q, D_emb)
            attn_weights: Attention weights, shape (..., H, N_q, N_kv)
        """
        
        # Get original shape
        orig_shape_q = query.shape
        orig_shape_k = key.shape
        
        # Flatten leading dimensions: (..., N, D) -> (B, N, D) where B = prod(...)
        # This allows us to handle arbitrary leading dimensions uniformly
        N_q = orig_shape_q[-2]
        N_k = orig_shape_k[-2]
        
        # Flatten to 3D
        query_flat = query.reshape(-1, N_q, self.embed_dim)  # (B, N_q, D_emb)
        key_flat = key.reshape(-1, N_k, self.embed_dim)  # (B, N_k, D_emb)
        value_flat = value.reshape(-1, N_k, self.embed_dim)  # (B, N_k, D_emb)
        if self.f_geo and dist_qk is not None:
            dist_qk_flat = dist_qk.reshape(-1, N_q, N_k, dist_qk.shape[-1])  # (B, N_q, N_k, D_r)
        else:
            dist_qk_flat = None
        
        B = query_flat.shape[0]
        
        if attn_mask is not None:
            attn_mask_flat = attn_mask.reshape(B, N_q, N_k)  # (B, N_q, N_k)
        else:
            attn_mask_flat = None
        
        # Equation 17-19: Compute Q, K, V
        Q = self.W_Q(query_flat)  # (B, N_q, D_emb)
        K = self.W_K(key_flat)  # (B, N_k, D_emb)
        V = self.W_V(value_flat)  # (B, N_k, D_emb)
        
        # Reshape for multi-head attention
        Q = Q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_q, head_dim)
        K = K.view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_k, head_dim)
        V = V.view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_k, head_dim)
        
        if self.sparse and attn_mask_flat is not None and not need_weights:
            # Use sparse attention implementation
            attn_output, attn_weights = self._sparse_attention(Q, K, V, attn_mask_flat, dist_qk_flat)  # (B, H, N_q, d_k)
        else:
            # Equation 20: Compute attention logits with geometric bias
            attn_output, attn_weights = self._dense_attention(Q, K, V, attn_mask_flat, dist_qk_flat, need_weights)  # (B, H, N_q, d_k)
            
        # combine heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, N_q, H, head_dim)
        attn_output = attn_output.view(B, N_q, self.embed_dim)  # (B, N_q, D_emb)
        attn_output = self.W_O(attn_output)  # (B, N_q, D_emb)
        attn_output = attn_output.view(*orig_shape_q[:-2], N_q, self.embed_dim)  # Restore original shape
        return attn_output, attn_weights   

    def _dense_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dist_qk: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, N_q, N_kv)
        
        # Equation 15: Compute geometry-aware attention bias
        if self.f_geo and dist_qk is not None:
            b_attn = self.f_geo(dist_qk)  # (B, N_q, N_kv, H)
            b_attn = b_attn.permute(0, 3, 1, 2)  # (B, H, N_q, N_kv)
        else:
            b_attn = 0
            
        attn_logits = attn_logits + b_attn  # Add geometric bias
        # Equation 21: Apply distance mask if provided
        if attn_mask is not None:
            # Convert boolean/binary mask to attention mask (-inf for masked positions)
            mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (B, H, N_q, N_kv)
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        
        # Equation 22: Compute attention weights
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, H, N_q, N_kv)
        attn_weights = self.dropout(attn_weights)
        
        # Equation 23: Apply attention to values
        out = torch.matmul(attn_weights, V)  # (B, H, N_q, d_k)
        if need_weights:
            return out, attn_weights
        else:
            return out, None
    
    def _sparse_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_mask: torch.Tensor,
        dist_qk: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sparse attention computation that only processes valid neighbor pairs.
        
        Fully vectorized implementation - no loops!
        More memory and compute efficient when the number of neighbors k << N_kv.
        
        Args:
            Q: Query tensor, shape (B, H, N_q, d_k)
            K: Key tensor, shape (B, H, N_kv, d_k)
            V: Value tensor, shape (B, H, N_kv, d_k)
            R_qkv: Geometric features, shape (B, N_q, N_kv, d_r)
            M_dist: Binary distance mask, shape (B, N_q, N_kv), 1=keep, 0=mask
        
        Returns:
            Output tensor, shape (B, H, N_q, d_k)
        """
        B, H, N_q, d_k = Q.shape
        _, _, N_kv, _ = K.shape
        # Get maximum number of neighbors (k_max)
        k_counts = attn_mask.sum(dim=-1)  # (B, N_q)
        k_max = int(k_counts.max().long().item())
        
        if k_max == 0:
            return torch.zeros(B, H, N_q, d_k, device=Q.device), None
        
        # Use topk to extract the indices of valid neighbors
        # Since attn_mask is binary, topk will get all the 1s (valid neighbors)
        _, neighbor_indices = torch.topk(attn_mask.float(), k=min(k_max, N_kv), dim=-1, largest=True)
        # neighbor_indices: (B, N_q, k_max)
        
        # Create mask for actually valid neighbors (since some queries may have < k_max neighbors)
        # Gather the mask values at selected indices to see which are actually valid
        neighbor_mask = torch.gather(attn_mask, 2, neighbor_indices) > 0  # (B, N_q, k_max)
        
        # Gather K and V for neighbors using advanced indexing
        # Expand indices for gathering: (B, H, N_q, k_max, d_k)
        idx_expanded = neighbor_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, N_q, k_max, d_k)
        
        # Gather K and V: (B, H, N_kv, d_k) -> (B, H, N_q, k_max, d_k)
        K_expanded = K.unsqueeze(2).expand(B, H, N_q, N_kv, d_k)
        V_expanded = V.unsqueeze(2).expand(B, H, N_q, N_kv, d_k)
        K_neighbors = torch.gather(K_expanded, 3, idx_expanded)
        V_neighbors = torch.gather(V_expanded, 3, idx_expanded)
        
        # Gather geometric bias: (B, H, N_q, N_kv) -> (B, H, N_q, k_max)
        # idx_bias = neighbor_indices.unsqueeze(1).expand(B, H, N_q, k_max)
        # bias_neighbors = torch.gather(b_attn, 3, idx_bias)
        
        # Compute attention scores: (B, H, N_q, k_max)
        # Q: (B, H, N_q, d_k) -> (B, H, N_q, 1, d_k)
        # K_neighbors: (B, H, N_q, k_max, d_k)
        attn_logits = torch.matmul(Q.unsqueeze(3), K_neighbors.transpose(-2, -1)).squeeze(3)  # (B, H, N_q, k_max)
        attn_logits = attn_logits / (d_k ** 0.5)
        
        # Compute geometry-aware attention bias for neighbors
        if self.f_geo and dist_qk is not None:
            idx_bias_expanded = neighbor_indices.unsqueeze(-1).expand(B, N_q, k_max, dist_qk.shape[-1])
            dist_qk_neighbors = torch.gather(
                dist_qk,
                2,
                idx_bias_expanded
            )  # (B, H, N_q, k_max, D_r)
            b_attn_neighbors = self.f_geo(dist_qk_neighbors)  # (B, H, N_q, k_max)
            b_attn_neighbors = b_attn_neighbors.permute(0, 3, 1, 2)  # (B, H, N_q, k_max)
        else:
            b_attn_neighbors = 0
        attn_logits = attn_logits + b_attn_neighbors
        attn_logits = attn_logits
        
        # Mask out invalid neighbors (those beyond actual k for each query)
        neighbor_mask_expanded = neighbor_mask.unsqueeze(1).expand(B, H, N_q, k_max)
        attn_logits = attn_logits.masked_fill(~neighbor_mask_expanded, float('-inf'))
        
        # Softmax over valid neighbors
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, H, N_q, k_max)
        attn_weights = attn_weights.masked_fill(~neighbor_mask_expanded, 0.0)  # Zero out invalid
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values: (B, H, N_q, k_max) x (B, H, N_q, k_max, d_k) -> (B, H, N_q, d_k)
        output = torch.matmul(attn_weights.unsqueeze(3), V_neighbors).squeeze(3)
        return output, None
    

class GeometryAwareEncoderLayer(nn.Module):
    """
    Geometry-aware Multi-Head Attention for neural downscaling.
    
    Implements equations (15)-(25) from the paper.
    
    Automatically handles inputs with arbitrary leading dimensions:
    - Single instance: (N_q, D)
    - Batched: (B, N_q, D)
    - Multi-dimensional: (B1, B2, ..., N_q, D)
    """
    
    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        d_geo_bias: Optional[int] = None,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "GELU",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        dim_geo_bias_feedforward: int = 64,
        sparse: bool = False,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            d_model: Dimension of query embeddings (D_emb)
            d_kv: Dimension of station tokens (D_kv)
            num_heads: Number of attention heads (H)
            d_geo_bias: Dimension of geometric features for bias (D_r)
            dropout: Dropout probability
            geo_bias_hidden_dim: Hidden dimension for geometry bias MLP (default: 64)
            ffn_expansion: Expansion ratio for FFN hidden layer (default: 4, standard in transformers)
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.self_attn = GeometryAwareMHA(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            d_geo_bias=d_geo_bias,  # Assuming geometric features have same dim as d_model
            geo_bias_hidden_dim=dim_geo_bias_feedforward,
            sparse=sparse,
            device=device,
            dtype=dtype,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = get_activation(activation)
        
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        dist_src: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of geometry-aware multi-head attention.
        
        Automatically handles arbitrary leading dimensions.
        
        Args:
            Z_q: Query embeddings, shape (..., N_q, D_emb)
            S_kv: Station tokens, shape (..., N_kv, D_kv)
            R_qkv: Relative geometric features, shape (..., N_q, N_kv, D_r)
            M_dist: Distance mask, shape (..., N_q, N_kv), optional
                   True/1 = keep, False/0 = mask out
        
        Returns:
            Z_tilde_q: Updated query embeddings, shape (..., N_q, D_emb)
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, dist_src)
            x = x + self._ffn_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, dist_src))
            x = self.norm2(x + self._ffn_block(x))
        return x
        
    def _sa_block(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        dist_src: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_output, _ = self.self_attn(
            x,
            x,
            x,
            dist_qk=dist_src,
            attn_mask=src_mask,
        )
        return self.dropout1(attn_output)
    
    def _ffn_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)