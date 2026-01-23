import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import copy
from torch.nn.init import xavier_uniform_

from oat.model_module.activations import get_activation

def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
class ObservationAssimilationTransformer(nn.Module):
    
    def __init__(
        self,
        *,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "GELU",
        layer_norm_eps: float = 1e-5,
        d_geo_bias: Optional[int] = None,
        dim_geo_bias_feedforward: int = 64,
        norm_first: bool = False,
        bias: bool = True,
        sparse_attn: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize a Geometry-Aware Transformer model.
        
        Args:
            d_model: The number of expected features in the encoder/decoder inputs. Defaults to 512.
            nhead: The number of heads in the multiheadattention models. Defaults to 8.
            num_encoder_layers: The number of sub-encoder-layers in the encoder. Defaults to 6.
            num_decoder_layers: The number of sub-decoder-layers in the decoder. Defaults to 6.
            dim_feedforward: The dimension of the feedforward network model. Defaults to 2048.
            dropout: The dropout value. Defaults to 0.1.
            activation: The activation function of intermediate layer. Defaults to "GELU".
            layer_norm_eps: The eps value in layer normalization components. Defaults to 1e-5.
            d_geo_bias: The dimension of the geometry bias. Defaults to None.
            dim_geo_bias_feedforward: The dimension of the feedforward network for geometry bias. Defaults to 64.
            norm_first: If True, layer norm is done prior to attention and feedforward operations. Defaults to False.
            bias: If True, adds bias to linear layers. Defaults to True.
            sparse_attn: If True, uses sparse attention mechanism. Defaults to False.
            device: The device to place the model on. Defaults to None.
            dtype: The desired data type of the model parameters. Defaults to None.
        """
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        encoder_layer = GeometryAwareEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            d_geo_bias=d_geo_bias,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            dim_geo_bias_feedforward=dim_geo_bias_feedforward,
            sparse=sparse_attn,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.encoder = GeometryAwareEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=encoder_norm,
        )
        
        decoder_layer = GeometryAwareDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            d_geo_bias=d_geo_bias,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            dim_geo_bias_feedforward=dim_geo_bias_feedforward,
            sparse=sparse_attn,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.decoder = GeometryAwareDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=decoder_norm,
        )
        
        self.d_model = d_model
        self.nhead = nhead
        self._reset_parameters()
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        dist_src: Optional[torch.Tensor] = None,
        dist_tgt: Optional[torch.Tensor] = None,
        dist_memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer model.
        Args:
            src (torch.Tensor): Source sequence tensor to be encoded.
            tgt (torch.Tensor): Target sequence tensor to be decoded.
            src_mask (Optional[torch.Tensor], optional): Mask for the source sequence to prevent 
                attention to certain positions. Defaults to None.
            tgt_mask (Optional[torch.Tensor], optional): Mask for the target sequence to prevent 
                attention to certain positions (e.g., for causal masking). Defaults to None.
            memory_mask (Optional[torch.Tensor], optional): Mask for the encoder output (memory) 
                in the decoder's cross-attention. Defaults to None.
            dist_src (Optional[torch.Tensor], optional): Distance information for the source sequence, 
                used for positional encoding or attention biasing. Defaults to None.
            dist_tgt (Optional[torch.Tensor], optional): Distance information for the target sequence, 
                used for positional encoding or attention biasing. Defaults to None.
            dist_memory (Optional[torch.Tensor], optional): Distance information for cross-attention 
                between target and encoder memory. Defaults to None.
        Returns:
            torch.Tensor: The output tensor from the decoder, representing the transformed target sequence.
        """
        memory = self.encoder(
            src,
            src_mask=src_mask,
            dist_src=dist_src,
        )
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            dist_tgt=dist_tgt,
            dist_memory=dist_memory,
        )
        return output
        
        
    def _reset_parameters(self) -> None:
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

class GeometryAwareEncoder(nn.Module):
    
    def __init__(
        self,
        *,
        encoder_layer: "GeometryAwareEncoderLayer",
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ):
        """
        Initialize the Geometry-Aware Transformer Encoder.
        Args:
            encoder_layer (GeometryAwareEncoderLayer): A single encoder layer instance that will be cloned
                to create the stack of encoder layers.
            num_layers (int): The number of encoder layers to stack.
            norm (Optional[nn.Module], optional): An optional normalization layer to apply after the
                final encoder layer. Defaults to None.
        Attributes:
            layers: A ModuleList containing cloned instances of the encoder layer.
            num_layers: The total number of encoder layers in the stack.
            norm: The normalization layer applied after all encoder layers, if provided.
        """
        
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        dist_src: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer encoder.
        Args:
            src (torch.Tensor): The input tensor to the transformer encoder.
                Shape: (batch_size, sequence_length, d_model)
            src_mask (Optional[torch.Tensor], optional): An optional mask tensor to prevent attention
                to certain positions. Shape: (batch_size, sequence_length, sequence_length) or
                broadcastable. Defaults to None.
            dist_src (Optional[torch.Tensor], optional): An optional distance tensor that can be used
                by the transformer layers for distance-aware attention or other mechanisms.
                Defaults to None.
        Returns:
            torch.Tensor: The output tensor after processing through all transformer layers.
                Shape: (batch_size, sequence_length, d_model)
        """
        output = src
        
        for mod in self.layers:
            output = mod(output, src_mask=src_mask, dist_src=dist_src)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output
    
class GeometryAwareDecoder(nn.Module):
    
    def __init__(
        self,
        *,
        decoder_layer: "GeometryAwareDecoderLayer",
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ):
        """
        Initialize the Transformer decoder.
        Args:
            decoder_layer: An instance of the decoder layer to be cloned.
            num_layers: Number of decoder layers to stack.
            norm: Normalization layer to apply after all decoder layers. 
                Defaults to None.
        Attributes:
            layers: List of cloned decoder layers.
            num_layers: Number of decoder layers.
            norm: Normalization layer.
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        dist_tgt: Optional[torch.Tensor] = None,
        dist_memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer decoder.
        
        Args:
            tgt: Target sequence tensor
            memory: Memory tensor from the encoder
            tgt_mask: Optional mask for target sequence
            memory_mask: Optional mask for memory tensor
            dist_tgt: Optional distance information for target sequence
            dist_memory: Optional distance information for memory
        Returns:
            Transformed output tensor after passing through all decoder layers and optional normalization
        """
        output = tgt
        
        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                dist_tgt=dist_tgt,
                dist_memory=dist_memory,
            )
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output
                 

class GeometryAwareEncoderLayer(nn.Module):
    
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
        Initialize a Transformer Encoder Layer with geometry-aware multi-head attention.
        Args:
            d_model: The number of expected features in the input.
            nhead: The number of heads in the multihead attention models.
            d_geo_bias: The dimension of geometric bias features. If None, no geometric bias is used.
            dim_feedforward: The dimension of the feedforward network model.
            dropout: The dropout value.
            activation: The activation function to use in the feedforward network.
            layer_norm_eps: The epsilon value in layer normalization components.
            norm_first: If True, layer norm is done prior to attention and feedforward operations (pre-norm). Otherwise it's done after (post-norm).
            dim_geo_bias_feedforward: The hidden dimension for the geometric bias feedforward network.
            sparse: Whether to use sparse attention.
            bias: Whether to use bias in linear layers and layer normalization.
            device: The device to place the parameters on.
            dtype: The data type for the parameters.
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
        Perform a forward pass through the transformer encoder layer.

        Args:
            src: The input tensor to the encoder layer.
            src_mask: An optional mask tensor for the source sequence to prevent attention to certain positions.
            dist_src: An optional distance tensor for the source sequence.

        Returns:
            The output tensor after applying self-attention and feed-forward operations.
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
    
    
class GeometryAwareDecoderLayer(nn.Module):
    
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
        Initialize a Transformer decoder layer with geometry-aware multi-head attention.
        Args:
            d_model: The number of expected features in the input.
            nhead: The number of heads in the multiheadattention models.
            d_geo_bias: Dimension of geometric bias features. If None, no geometric bias is used.
            dim_feedforward: The dimension of the feedforward network model.
            dropout: The dropout value.
            activation: The activation function of the intermediate layer.
            layer_norm_eps: The eps value in layer normalization components.
            norm_first: If True, layer norm is done prior to self attention, multihead attention and feedforward operations.
            dim_geo_bias_feedforward: Hidden dimension for geometric bias feedforward network.
            sparse: Whether to use sparse attention.
            bias: Whether to use bias in linear layers and layer normalization.
            device: The device to place the parameters on.
            dtype: The dtype of the parameters.
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
        self.multihead_attn = GeometryAwareMHA(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            d_geo_bias=d_geo_bias,
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
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = get_activation(activation)
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        dist_tgt: Optional[torch.Tensor] = None,
        dist_memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the transformer decoder layer.

        Args:
            tgt: The target sequence tensor
            memory: The memory sequence tensor from the encoder
            tgt_mask: Optional mask for the target sequence
            memory_mask: Optional mask for the memory sequence
            dist_tgt: Optional distance information for the target sequence
            dist_memory: Optional distance information for the memory sequence

        Returns:
            The output tensor after applying self-attention, cross-attention, and feedforward blocks
        """
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, dist_tgt)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, dist_memory)
            x = x + self._ffn_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, dist_tgt))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, dist_memory))
            x = self.norm3(x + self._ffn_block(x))
        return x
    
    
    def _sa_block(
        self,
        x: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        dist_tgt: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_output, _ = self.self_attn(
            x,
            x,
            x,
            dist_qk=dist_tgt,
            attn_mask=tgt_mask,
        )
        return self.dropout1(attn_output)
    
    def _mha_block(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
        dist_memory: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_output, _ = self.multihead_attn(
            x,
            memory,
            memory,
            dist_qk=dist_memory,
            attn_mask=memory_mask,
        )
        return self.dropout2(attn_output)
    
    def _ffn_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
    
    
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
        Initialize the Transformer attention module.
        Args:
            embed_dim: The embedding dimension for the transformer.
            num_heads: The number of attention heads.
            dropout: Dropout probability for attention weights. Defaults to 0.0.
            d_geo_bias: Dimension of geometric bias features. If None, no geometric bias is used.
            geo_bias_hidden_dim: Hidden dimension for the geometric bias network. Defaults to 64.
            sparse: Whether to use sparse attention. Defaults to False.
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
                nn.ReLU(),
                nn.Linear(num_heads * geo_bias_hidden_dim, num_heads, **factory_kwargs)
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
    