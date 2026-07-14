import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import copy
from torch.nn.init import xavier_uniform_

from oat.model_module.activations import get_activation
from oat.model_module.attention import GeometryAwareMHA

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
        activation_geo_bias: str = "GELU",
        norm_first: bool = False,
        bias: bool = True,
        sparse_attn: bool = False,
        decoder_self_attention: bool = False,
        geo_attn_mode: str = "additive",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize a Geometry-Aware Transformer model.

        Args:
            d_model: The number of expected features in the encoder/decoder inputs. Defaults to 512.
            nhead: The number of heads in the multi-head attention layers. Defaults to 8.
            num_encoder_layers: The number of sub-encoder-layers in the encoder. Defaults to 6.
            num_decoder_layers: The number of sub-decoder-layers in the decoder. Defaults to 6.
            dim_feedforward: The dimension of the feedforward network model. Defaults to 2048.
            dropout: The dropout value. Defaults to 0.1.
            activation: The activation function of intermediate layer. Defaults to "GELU".
            layer_norm_eps: The eps value in layer normalization components. Defaults to 1e-5.
            d_geo_bias: The dimension of the geometry bias. Defaults to None.
            dim_geo_bias_feedforward: The dimension of the feedforward network for geometry bias. Defaults to 64.
            activation_geo_bias: The activation function for geometry bias feedforward network. Defaults to "ReLU".
            bias: If True, adds bias to linear layers. Defaults to True.
            sparse_attn: If True, uses sparse attention mechanism. Defaults to False.
            decoder_self_attention: If True, the decoder will perform self-attention.
            geo_attn_mode: The geometry-aware attention mode ("additive", "tetnp", or "vanilla"). Defaults to "additive".
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
            activation_geo_bias=activation_geo_bias,
            sparse_attn=sparse_attn,
            geo_attn_mode=geo_attn_mode,
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
            activation_geo_bias=activation_geo_bias,
            sparse_attn=sparse_attn,
            geo_attn_mode=geo_attn_mode,
            bias=bias,
            self_attention=decoder_self_attention,
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
        src_mask_edges: Optional[torch.Tensor] = None,
        tgt_mask_edges: Optional[torch.Tensor] = None,
        memory_mask_edges: Optional[torch.Tensor] = None,
        g_src: Optional[torch.Tensor] = None,
        g_tgt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer model.
        Args:
            src (torch.Tensor): Source sequence tensor to be encoded.
            tgt (torch.Tensor): Target sequence tensor to be decoded.
            src_mask_edges (Optional[torch.Tensor], optional): Edges for the source sequence to prevent 
                attention to certain positions. Defaults to None.
            tgt_mask_edges (Optional[torch.Tensor], optional): Edges for the target sequence to prevent 
                attention to certain positions (e.g., for causal masking). Defaults to None.
            memory_mask_edges (Optional[torch.Tensor], optional): Edges for the encoder output (memory) 
                in the decoder's cross-attention. Defaults to None.
            g_src (Optional[torch.Tensor], optional): Geometry information for the source sequence, 
                used for positional encoding or attention biasing. Defaults to None.
            g_tgt (Optional[torch.Tensor], optional): Geometry information for the target sequence, 
                used for positional encoding or attention biasing. Defaults to None.
        Returns:
            torch.Tensor: The output tensor from the decoder, representing the transformed target sequence.
        """
        memory = self.encoder(
            src,
            src_mask_edges=src_mask_edges,
            g_src=g_src,
        )
        output = self.decoder(
            tgt,
            memory,
            tgt_mask_edges=tgt_mask_edges,
            memory_mask_edges=memory_mask_edges,
            g_tgt=g_tgt,
            g_src=g_src,
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
        src_mask_edges: Optional[torch.Tensor] = None,
        g_src: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer encoder.
        Args:
            src (torch.Tensor): The input tensor to the transformer encoder.
                Shape: (batch_size, sequence_length, d_model)
            src_mask_edges (Optional[torch.Tensor], optional): An optional mask edges tensor to prevent attention
                to certain positions. Shape: (batch_size, sequence_length, sequence_length) or
                broadcastable. Defaults to None.
            g_src (Optional[torch.Tensor], optional): An optional geometry tensor that can be used
                by the transformer layers for geometry-aware attention or other mechanisms.
                Defaults to None.
        Returns:
            torch.Tensor: The output tensor after processing through all transformer layers.
                Shape: (batch_size, sequence_length, d_model)
        """
        output = src
        
        for mod in self.layers:
            output = mod(output, src_mask_edges=src_mask_edges, g_src=g_src)
        
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
        tgt_mask_edges: Optional[torch.Tensor] = None,
        memory_mask_edges: Optional[torch.Tensor] = None,
        g_tgt: Optional[torch.Tensor] = None,
        g_src: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer decoder.
        
        Args:
            tgt: Target sequence tensor
            memory: Memory tensor from the encoder
            tgt_mask_edges: Optional mask edges for target sequence
            memory_mask_edges: Optional mask edges for memory tensor
            g_tgt: Optional geometry information for target sequence
            g_src: Optional geometry information for memory
        Returns:
            Transformed output tensor after passing through all decoder layers and optional normalization
        """
        output = tgt
        
        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask_edges=tgt_mask_edges,
                memory_mask_edges=memory_mask_edges,
                g_tgt=g_tgt,
                g_src=g_src,
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
        activation_geo_bias: str = "GELU",
        sparse_attn: bool = False,
        geo_attn_mode: str = "additive",
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
            activation_geo_bias: The activation function for the geometry bias feedforward network.
            sparse_attn: Whether to use sparse attention.
            geo_attn_mode: The geometry-aware attention mode ("additive", "tetnp", or "vanilla"). Defaults to "additive".
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
            activation_geo_bias=activation_geo_bias,
            sparse_attn=sparse_attn,
            geo_attn_mode=geo_attn_mode,
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
        src_mask_edges: Optional[torch.Tensor] = None,
        g_src: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the transformer encoder layer.

        Args:
            src: The input tensor to the encoder layer.
            src_mask_edges: An optional mask tensor for the source sequence to prevent attention to certain positions.
            g_src: An optional geometry tensor for the source sequence.

        Returns:
            The output tensor after applying self-attention and feed-forward operations.
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask_edges, g_src)
            x = x + self._ffn_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask_edges, g_src))
            x = self.norm2(x + self._ffn_block(x))
        return x
        
    def _sa_block(
        self,
        x: torch.Tensor,
        src_mask_edges: Optional[torch.Tensor],
        g_src: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_output, _ = self.self_attn(
            x,
            x,
            x,
            geo_q=g_src,
            geo_k=g_src,
            attn_edges=src_mask_edges,
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
        activation_geo_bias: str = "GELU",
        sparse_attn: bool = False,
        geo_attn_mode: str = "additive",
        bias: bool = True,
        self_attention: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize a Transformer decoder layer with geometry-aware multi-head attention.
        Args:
            d_model: The number of expected features in the input.
            nhead: The number of heads in the multi-head attention models.
            d_geo_bias: Dimension of geometric bias features. If None, no geometric bias is used.
            dim_feedforward: The dimension of the feedforward network model.
            dropout: The dropout value.
            activation: The activation function of the intermediate layer.
            layer_norm_eps: The eps value in layer normalization components.
            norm_first: If True, layer norm is done prior to self attention, multihead attention and feedforward operations.
            dim_geo_bias_feedforward: Hidden dimension for geometric bias feedforward network.
            activation_geo_bias: The activation function for the geometry bias feedforward network.
            sparse_attn: Whether to use sparse attention.
            geo_attn_mode: The geometry-aware attention mode ("additive", "tetnp", or "vanilla"). Defaults to "additive".
            bias: Whether to use bias in linear layers and layer normalization.
            self_attention: whether self-attention is performed.
            device: The device to place the parameters on.
            dtype: The dtype of the parameters.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.perform_self_attention = self_attention
        if self_attention:
            self.self_attn = GeometryAwareMHA(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                d_geo_bias=d_geo_bias,  # Assuming geometric features have same dim as d_model
                geo_bias_hidden_dim=dim_geo_bias_feedforward,
                activation_geo_bias=activation_geo_bias,
                sparse_attn=sparse_attn,
                geo_attn_mode=geo_attn_mode,
                device=device,
                dtype=dtype,
            )
        self.multihead_attn = GeometryAwareMHA(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            d_geo_bias=d_geo_bias,
            geo_bias_hidden_dim=dim_geo_bias_feedforward,
            activation_geo_bias=activation_geo_bias,
            sparse_attn=sparse_attn,
            geo_attn_mode=geo_attn_mode,
            device=device,
            dtype=dtype,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        
        self.norm_first = norm_first
        
        if self_attention:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        if self_attention:
            self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = get_activation(activation)
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask_edges: Optional[torch.Tensor] = None,
        memory_mask_edges: Optional[torch.Tensor] = None,
        g_tgt: Optional[torch.Tensor] = None,
        g_src: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the transformer decoder layer.

        Args:
            tgt: The target sequence tensor
            memory: The memory sequence tensor from the encoder
            tgt_mask_edges: Optional mask for the target sequence
            memory_mask_edges: Optional mask for the memory sequence
            g_tgt: Optional geometry information for the target sequence
            g_src: Optional geometry information for the memory sequence

        Returns:
            The output tensor after applying self-attention, cross-attention, and feedforward blocks
        """
        x = tgt
        if self.norm_first:
            
            if self.perform_self_attention:
                x = x + self._sa_block(self.norm1(x), tgt_mask_edges, g_tgt)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask_edges, g_src, g_tgt)
            x = x + self._ffn_block(self.norm3(x))
        else:
            if self.perform_self_attention:
                x = self.norm1(x + self._sa_block(x, tgt_mask_edges, g_tgt))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask_edges, g_src, g_tgt))
            x = self.norm3(x + self._ffn_block(x))
        return x
    
    
    def _sa_block(
        self,
        x: torch.Tensor,
        tgt_edges: Optional[torch.Tensor],
        g_tgt: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_output, _ = self.self_attn(
            x,
            x,
            x,
            geo_q=g_tgt,
            geo_k=g_tgt,
            attn_edges=tgt_edges,
        )
        return self.dropout1(attn_output)
    
    def _mha_block(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_edges: Optional[torch.Tensor],
        g_src: Optional[torch.Tensor],
        g_tgt: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_output, _ = self.multihead_attn(
            x,
            memory,
            memory,
            geo_q=g_tgt,
            geo_k=g_src,
            attn_edges=memory_edges,
        )
        return self.dropout2(attn_output)
    
    def _ffn_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)