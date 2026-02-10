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
        activation_geo_bias: str = "ReLU",
        sparse: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the Transformer attention module.
        Args:
            embed_dim: The embedding dimension for the transformer.
            num_heads: The number of attention heads.
            dropout: Dropout probability for attention weights. Defaults to 0.0.
            d_geo_bias: Dimension of the input geometric feature vector used to compute per-head attention bias. If None, no geometric bias is used.
            geo_bias_hidden_dim: Hidden dimension for the geometric bias network. Defaults to 64.
            activation_geo_bias: Activation function for the geometric bias network. Defaults to "ReLU".
            sparse: Whether to use sparse attention. Checks also if mask is defined. Defaults to False.
            device: Device to place tensors on.
            dtype: Data type for tensors.
        Raises:
            ValueError: If embed_dim or num_heads are not greater than 0.
            AssertionError: If embed_dim is not divisible by num_heads.
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
                nn.Linear(d_geo_bias, num_heads * geo_bias_hidden_dim, **factory_kwargs),
                get_activation(activation_geo_bias),
                nn.Linear(num_heads * geo_bias_hidden_dim, num_heads, **factory_kwargs)
            )
        else:
            self.f_geo = None
        
        self.dropout = nn.Dropout(dropout)
        # Output projection layer for combining attention heads
        self.W_O = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        
        
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
        Compute multi-head attention with optional geometric bias and sparse attention.
        This method implements the forward pass of the attention mechanism, supporting:
        - Multi-head attention with learnable projections
        - Optional geometric/distance-based bias
        - Optional sparse attention for efficiency
        - Arbitrary leading batch dimensions
        Args:
            query: Query tensor of shape (..., N_q, D_emb) where ... represents arbitrary leading dimensions
            key: Key tensor of shape (..., N_k, D_emb)
            value: Value tensor of shape (..., N_k, D_emb)
            dist_qk: Optional distance/geometric features between queries and keys of shape (..., N_q, N_k, d_geo_bias)
            need_weights: Whether to return attention weights. If False and sparse attention is enabled, 
                          weights may not be computed for efficiency
            attn_mask: Optional attention mask of shape (..., N_q, N_k) where True/1 indicates positions 
                       to attend to
        Returns:
            A tuple containing:
                - attn_output: Attention output tensor of shape (..., N_q, D_emb), same leading dimensions as query
                - attn_weights: Attention weights of shape (B, H, N_q, N_k) if computed, otherwise None
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
        if self.f_geo is not None and dist_qk is not None:
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
        attn_mask: Optional[torch.Tensor] = None,
        dist_qk: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute dense (full) attention between query, key, and value tensors.
        This method implements the standard scaled dot-product attention mechanism with optional
        geometry-aware bias and masking. It follows equations 15, 21, 22, and 23 from the paper.
        Args:
            Q: Query tensor of shape (B, H, N_q, d_k) where B is batch size, H is number of heads,
                N_q is number of queries, and d_k is head dimension
            K: Key tensor of shape (B, H, N_kv, d_k)
            V: Value tensor of shape (B, H, N_kv, d_k)
            attn_mask: Optional boolean/binary mask of shape (B, N_q, N_kv) where 0 indicates
                positions to mask out. Defaults to None
            dist_qk: Optional distance tensor of shape (B, N_q, N_kv, d_geo_bias) for computing
                geometry-aware attention bias. Defaults to None
            need_weights: Whether to return attention weights. Defaults to False
        Returns:
            A tuple containing:
                - Output tensor of shape (B, H, N_q, d_k) with attended values
                - Attention weights of shape (B, H, N_q, N_kv) if need_weights is True, else None
        """
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
        # Replace any NaNs (e.g., from fully masked rows) with zeros to avoid propagating NaNs
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
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
        Compute sparse attention by only considering valid neighbors specified by attention mask.
        This method implements an efficient sparse attention mechanism that:
        1. Identifies valid neighbors for each query using the attention mask
        2. Extracts only the relevant key-value pairs for computation
        3. Applies scaled dot-product attention with optional geometric bias
        4. Returns the attention output for valid neighbors only
        Args:
            Q: Query tensor of shape (batch_size, num_heads, num_queries, head_dim)
            K: Key tensor of shape (batch_size, num_heads, num_keys, head_dim)
            V: Value tensor of shape (batch_size, num_heads, num_values, head_dim)
            attn_mask: Binary attention mask of shape (batch_size, num_queries, num_keys) 
                indicating valid connections (1 for valid, 0 for invalid)
            dist_qk: Optional distance tensor for geometric bias computation
        Returns:
            A tuple containing:
                - output: Attention output tensor of shape (batch_size, num_heads, num_queries, head_dim)
                - None: Placeholder for attention weights (not returned in sparse implementation)
        """
        assert attn_mask is not None, "Attention mask must be provided for sparse attention."
        B, H, N_q, d_k = Q.shape
        _, _, N_kv, _ = K.shape
        # Get maximum number of neighbors (k_max)
        k_counts = attn_mask.sum(dim=-1)  # (B, N_q)
        k_max = int(k_counts.max().long().item())
        
        if k_max == 0:
            # No valid neighbors for any query: return zeros.
            # Note: this early return bypasses dropout used in the normal path.
            #       Dropout on an all-zero tensor is a no-op, so skipping it here
            #       does not change the output or gradients; this behavior is
            #       intentional and documents the edge case explicitly.
            #       However, because dropout normally consumes random numbers from
            #       the global RNG, this shortcut does not advance the RNG state
            #       in the same way as the full path. As a result, subsequent
            #       stochastic operations that rely on the global RNG may differ
            #       between runs with and without valid neighbors. If strict,
            #       bitwise reproducibility of RNG-dependent behavior is required,
            #       callers should account for this difference.
            return torch.zeros(B, H, N_q, d_k, device=Q.device, dtype=Q.dtype), None
        
        # Use topk to extract the indices of valid neighbors.
        # To ensure deterministic behavior when multiple positions share the same
        # mask value (binary 0/1), build a score tensor that:
        #   - ranks all valid positions (mask==1) above all invalid ones (mask==0)
        #   - breaks ties among valid positions by favoring lower key indices.
        idx = torch.arange(N_kv, device=attn_mask.device).view(1, 1, N_kv)
        idx = idx.expand_as(attn_mask)
        # Valid positions get scores in (0, N_kv+1], invalid positions get <= 0.
        scores = attn_mask * (N_kv + 1) - idx
        _, neighbor_indices = torch.topk(scores, k=min(k_max, N_kv), dim=-1, largest=True)
        # neighbor_indices: (B, N_q, k_max)
        
        # Create mask for actually valid neighbors (since some queries may have < k_max neighbors)
        # Gather the mask values at selected indices to see which are actually valid
        neighbor_mask = torch.gather(attn_mask, 2, neighbor_indices) > 0  # (B, N_q, k_max)
        
        # Gather K and V for neighbors using torch.gather along the N_kv dimension
        # neighbor_indices: (B, N_q, k_max) indexing the N_kv dimension
        N_kv = K.size(2)
        # Build gather index tensor over N_kv, expanded to (B, H, N_q, k_max, d_k)
        kv_idx_gather = neighbor_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, N_q, k_max, d_k)
        # Expand K, V to include the query dimension before gathering
        K_expanded = K.unsqueeze(2).expand(B, H, N_q, N_kv, d_k)
        V_expanded = V.unsqueeze(2).expand(B, H, N_q, N_kv, d_k)
        K_neighbors = torch.gather(K_expanded, 3, kv_idx_gather)  # (B, H, N_q, k_max, d_k)
        V_neighbors = torch.gather(V_expanded, 3, kv_idx_gather)  # (B, H, N_q, k_max, d_k)
        # Gather geometric bias: (B, H, N_q, N_kv) -> (B, H, N_q, k_max)
        
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
            )  # (B, N_q, k_max, D_r)
            # self.f_geo maps per-query, per-neighbor distances to per-head bias
            b_attn_neighbors = self.f_geo(dist_qk_neighbors)  # (B, N_q, k_max, H)
            b_attn_neighbors = b_attn_neighbors.permute(0, 3, 1, 2)  # (B, H, N_q, k_max)
        else:
            b_attn_neighbors = 0
        attn_logits = attn_logits + b_attn_neighbors
        
        # Mask out invalid neighbors (those beyond actual k for each query)
        neighbor_mask_expanded = neighbor_mask.unsqueeze(1).expand(B, H, N_q, k_max)
        attn_logits = attn_logits.masked_fill(~neighbor_mask_expanded, float('-inf'))
        
        # Softmax over valid neighbors
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, H, N_q, k_max)
        attn_weights = attn_weights.masked_fill(~neighbor_mask_expanded, 0.0)  # Zero out invalid
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values: (B, H, N_q, k_max) x (B, H, N_q, k_max, d_k) -> (B, H, N_q, d_k)
        output = torch.matmul(attn_weights.unsqueeze(3), V_neighbors).squeeze(3)
        return output, None