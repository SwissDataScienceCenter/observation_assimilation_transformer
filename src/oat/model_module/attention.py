import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from oat.model_module.activations import get_activation
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter

class GeometryAwareMHA(nn.Module):
    
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        d_geo_bias: Optional[int] = None,
        geo_bias_hidden_dim: int = 64,
        activation_geo_bias: str = "GELU",
        sparse_attn: bool = False,
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
            activation_geo_bias: Activation function for the geometric bias network. Defaults to "GELU".
            sparse_attn: Whether to use sparse attention. Checks also if mask is defined. Defaults to False.
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
        self.sparse_attn = sparse_attn
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
        geo_k: Optional[torch.Tensor] = None,
        geo_q: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_edges: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute multi-head attention with optional geometric bias and sparse attention.
        This method implements the forward pass of the attention mechanism, supporting:
        - Multi-head attention with learnable projections
        - Optional geometric/distance-based bias
        - Optional sparse attention for efficiency
        - Arbitrary leading batch dimensions
        Args:
            query: Query tensor of shape (B, N_q, D_emb)
            key: Key tensor of shape (B, N_k, D_emb)
            value: Value tensor of shape (B, N_k, D_emb)
            geo_k: Optional geometric features for keys of shape (B, N_k, d_geo_bias)
            geo_q: Optional geometric features for queries of shape (B, N_q, d_geo_bias)
            need_weights: Whether to return attention weights. If False and sparse attention is enabled, 
                          weights may not be computed for efficiency
            attn_edges: Optional attention edge indices of shape [2, E] in flattened batch space
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
        B = orig_shape_q[0]
        N_q = orig_shape_q[-2]
        N_k = orig_shape_k[-2]
        
        # Equation 17-19: Compute Q, K, V
        Q = self.W_Q(query)  # (B, N_q, D_emb)
        K = self.W_K(key)  # (B, N_k, D_emb)
        V = self.W_V(value)  # (B, N_k, D_emb)
        
        Q_heads = Q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_q, d_k)
        K_heads = K.view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_k, d_k)
        V_heads = V.view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_k, d_k)

        if not self.sparse_attn or attn_edges is None:
            # Dense attention path
            
            out, attn_weights_out = self._dense_attention(
                Q_heads,  # (B, H, N_q, d_k)
                K_heads,  # (B, H, N_k, d_k)
                V_heads,  # (B, H, N_k, d_k)
                attn_edges=attn_edges,  # (2, E) or None
                geo_q=geo_q,  # (B, N_q, d_geo_bias) or None
                geo_k=geo_k,  # (B, N_k, d_geo_bias) or None
                need_weights=need_weights
            )
            
        else:
            out, attn_weights_out = self._sparse_attention(
                Q_heads,  # (B, H, N_q, d_k)
                K_heads,  # (B, H, N_k, d_k)
                V_heads,  # (B, H, N_k, d_k)
                attn_edges=attn_edges,  # (2, E)
                geo_q=geo_q,  # (B, N_q, d_geo_bias) or None
                geo_k=geo_k,  # (B, N_k, d_geo_bias) or None
                need_weights=need_weights
            )
            

        # combine heads
        out = out.transpose(1, 2).contiguous()  # (B, N_q, H, d_k)
        out = out.view(B, N_q, self.embed_dim)  # (B, N_q, D_emb)
        attn_output = self.W_O(out)  # (B, N_q, D_emb)
        attn_output = attn_output.view(B, N_q, self.embed_dim)  # Restore original shape
        return attn_output, attn_weights_out
    
    def _dist_function(self, geo_q: torch.Tensor, geo_k: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between query and key geometric features.
        Args:
            geo_q: Geometric features for queries of shape (B, N_q, d_geo)
            geo_k: Geometric features for keys of shape (B, N_k, d_geo)
        Returns:
            A tensor of pairwise distances of shape (B, N_q, N_k, d_geo)
        """
        dist_qk = geo_q - geo_k  # (B, N_q, N_k, d_geo)
        return dist_qk
    
    def _sparse_attention(
        self,
        Q: torch.Tensor,  # [B, H, N_q, d_k]
        K: torch.Tensor,  # [B, H, N_k, d_k]
        V: torch.Tensor,  # [B, H, N_k, d_k]
        attn_edges: torch.Tensor,  # [2, E]
        geo_q: Optional[torch.Tensor] = None,
        geo_k: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, H, N_q, d_k = Q.shape
        N_k = K.shape[2]
        
        # Flatten batch dimension
        Q_flat = Q.transpose(1, 2).reshape(B * N_q, H, d_k)  # [B*N_q, H, d_k]
        K_flat = K.transpose(1, 2).reshape(B * N_k, H, d_k)  # [B*N_k, H, d_k]
        V_flat = V.transpose(1, 2).reshape(B * N_k, H, d_k)  # [B*N_k, H, d_k]
        
        # Extract edge features
        edge_Q = Q_flat[attn_edges[0]]  # [E, H, d_k]
        edge_K = K_flat[attn_edges[1]]  # [E, H, d_k]
        edge_V = V_flat[attn_edges[1]]  # [E, H, d_k]
        
        # Equation 20: Scaled dot-product attention
        attn_logits = (edge_Q * edge_K).sum(dim=-1) / (d_k ** 0.5)  # [E, H]
        
        # Equation 15: Geometry bias
        if self.f_geo is not None and geo_k is not None and geo_q is not None:
            geo_q_flat = geo_q.reshape(B * N_q, -1)  # [B*N_q, d_geo]
            geo_k_flat = geo_k.reshape(B * N_k, -1)  # [B*N_k, d_geo]
            
            dist_qk = self._dist_function(geo_q_flat[attn_edges[0]], geo_k_flat[attn_edges[1]])  # [E, d_geo]
            b_attn = self.f_geo(dist_qk)  # [E, H]
            attn_logits = attn_logits + b_attn
        
        # Equation 22: Softmax per query node
        attn_weights = pyg_softmax(attn_logits, attn_edges[0], num_nodes=B * N_q)  # [E, H]
        attn_weights = self.dropout(attn_weights)
        
        # Equation 23: Aggregate values
        weighted_V = attn_weights.unsqueeze(-1) * edge_V  # [E, H, d_k]
        out = scatter(weighted_V, attn_edges[0], dim=0, dim_size=B * N_q, reduce='sum')  # [B*N_q, H, d_k]
        out = out.view(B, N_q, H, d_k).transpose(1, 2)  # [B, H, N_q, d_k]
        
        # Optionally reconstruct dense weights (expensive!)
        if need_weights:
            attn_weights_dense = torch.zeros(B * N_q, N_k, H, device=Q.device)
            attn_weights_dense[attn_edges[0], attn_edges[1]] = attn_weights
            attn_weights_out = attn_weights_dense.view(B, N_q, N_k, H).permute(0, 3, 1, 2)  # [B, H, N_q, N_k]
        else:
            attn_weights_out = None
        
        return out, attn_weights_out

    def _dense_attention(
        self,
        Q: torch.Tensor,  # [B, H, N_q, d_k]
        K: torch.Tensor,  # [B, H, N_k, d_k]
        V: torch.Tensor,  # [B, H, N_k, d_k]
        attn_edges: Optional[torch.Tensor] = None,
        geo_q: Optional[torch.Tensor] = None,
        geo_k: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, H, N_q, d_k = Q.shape
        N_k = K.shape[2]
        
        # Equation 20: QK^T / sqrt(d_k)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # [B, H, N_q, N_k]
        
        # Equation 15: Geometry bias
        if self.f_geo is not None and geo_k is not None and geo_q is not None:
            # Compute pairwise geometry features
            geo_q_exp = geo_q.unsqueeze(2)  # [B, N_q, 1, d_geo]
            geo_k_exp = geo_k.unsqueeze(1)  # [B, 1, N_k, d_geo]
            dist_qk = self._dist_function(geo_q_exp, geo_k_exp)  # [B, N_q, N_k, d_geo]
            
            b_attn = self.f_geo(dist_qk)  # [B, N_q, N_k, H]
            b_attn = b_attn.permute(0, 3, 1, 2)  # [B, H, N_q, N_k]
            attn_logits = attn_logits + b_attn
        
        # Equation 21: Apply distance mask
        if attn_edges is not None:
            mask = torch.zeros(B, N_q, N_k, dtype=torch.bool, device=Q.device)
            # Unflatten edges: assume edges are in flattened [0, B*N_q) and [0, B*N_k)
            batch_idx = attn_edges[0] // N_q
            q_idx = attn_edges[0] % N_q
            k_idx = attn_edges[1] % N_k
            mask[batch_idx, q_idx, k_idx] = True
            
            mask = mask.unsqueeze(1)  # [B, 1, N_q, N_k]
            attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
        
        # Equation 22: Softmax
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        
        # Equation 23: Apply to values
        out = torch.matmul(attn_weights, V)  # [B, H, N_q, d_k]
        
        return out, attn_weights if need_weights else None


    def _from_edges_to_mask(self, attn_edge_index: torch.Tensor, B: int, N_q: int, N_k: int) -> torch.Tensor:
        """Convert edge indices to dense mask."""
        mask = torch.zeros(B, N_q, N_k, dtype=torch.bool, device=attn_edge_index.device)
        
        # Unflatten batch dimension from edges
        batch_idx = attn_edge_index[0] // N_q
        q_idx = attn_edge_index[0] % N_q
        k_idx = attn_edge_index[1] % N_k
        
        mask[batch_idx, q_idx, k_idx] = True
        return mask