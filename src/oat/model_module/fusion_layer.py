import torch
import torch.nn as nn
from oat.model_module.activations import get_activation
from abc import abstractmethod

class FiLMLayer(nn.Module):
    """
    FiLM for tabular data [..., C].
    
    Applies: γ * x + β
    
    Args:
        num_features: Feature dimension C
        cond_dim: Conditioning dimension (e.g., time features)
        hidden_dim: MLP hidden size (default: num_features)
    """
    
    def __init__(self, 
                 num_features: int, 
                 cond_dim: int, 
                 hidden_dim: int | None = None,
                 activation: str = "SiLU"):
        super().__init__()
        
        act_fn = get_activation(activation)
        if hidden_dim is None:
            hidden_dim = 2 * num_features
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 2 * num_features)
        )
        
        # Identity initialization
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.mlp[-1].bias.data[:num_features] = 1.0  # gamma = 1
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., C] - features
            cond: [..., cond_dim] or [cond_dim] - conditioning (broadcasted if needed)
        
        Returns:
            [..., C] - modulated features
        """
        # Compute gamma and beta
        params = self.mlp(cond)  # [..., 2*C]
        gamma, beta = params.chunk(2, dim=-1)  # [..., C] each
        
        return gamma * x + beta
    
    
class AdaLNLayer(nn.Module):
    """
    Adaptive LayerNorm for tabular data [..., C].
    
    Applies: γ * normalize(x) + β
    
    Args:
        num_features: Feature dimension C
        cond_dim: Conditioning dimension
        eps: Numerical stability constant
    """
    
    def __init__(self, 
                 num_features: int, 
                 cond_dim: int, 
                 activation: str = "SiLU",
                 eps: float = 1e-5):
        super().__init__()
        
        self.eps = eps
        self.ln = nn.LayerNorm(num_features, eps=eps, elementwise_affine=False)
        
        act_fn = get_activation(activation)
        self.mlp = nn.Sequential(
            act_fn,
            nn.Linear(cond_dim, 2 * num_features)
        )
        # Identity initialization
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.mlp[-1].bias.data[:num_features] = 1.0
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., C] - features
            cond: [..., cond_dim] or [cond_dim] - conditioning
        
        Returns:
            [..., C] - normalized and modulated features
        """
        # Normalize
        x_norm = self.ln(x)
        
        # Compute modulation
        params = self.mlp(cond)
        gamma, beta = params.chunk(2, dim=-1)
        
        return gamma * x_norm + beta


class BaseFusionLayer(nn.Module):
    """
    Base class for fusion layers that combine multiple input tensors into a single output tensor.
    Attributes:
        in_channels: List of input channel counts for each input tensor.
        out_channels: Number of output channels for the fused tensor.
    """
    
    def __init__(self, in_channels: list[int], out_channels: int):
        super().__init__()

        self.in_channels = torch.Tensor(in_channels)
        self.out_channels = out_channels
        
    @abstractmethod
    def forward(self, *inputs: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        pass
    
    
class FusionBlock(nn.Module):
    """
    FusionBlock: A modular fusion layer for conditional feature processing.
    This block combines normalization and feature mixing with optional residual learning.
    It supports multiple normalization strategies (FiLM, AdaLN, LayerNorm) that can be
    applied before or after the mixing operation.
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        cond_channels: Number of conditioning channels (for FiLM/AdaLN)
        norm_first: If True, apply normalization before mixing; otherwise after
        norm_layer: Type of normalization layer - "FiLM", "AdaLN", or "LayerNorm"
        dropout_p: Dropout probability applied in the mixing layer
        residual_learning: If True, adds input to output (identity + transformation)
    Attributes:
        norm: The normalization module selected based on norm_layer parameter
        mixing_layer: Sequential module containing linear transformation with GLU activation
                      and optional dropout
        norm_first: Flag indicating normalization order
        residual_learning: Flag indicating whether to use residual connections
    Returns:
        Fused feature tensor with shape matching output channels
    Raises:
        ValueError: If norm_layer is not one of the supported types
    """
    
    norm: nn.Module
    
    def __init__(self,
                in_channels: int,
                out_channels: int,
                cond_channels: int,
                norm_first: bool = True,
                norm_layer: str = "AdaLN",
                dropout_p: float = 0.0,
                residual_learning: bool = False):
        
        super().__init__()
        self.norm_first = norm_first
        self.residual_learning = residual_learning
        
        ln_in_dim = in_channels if norm_first else out_channels
    
        if norm_layer == "FiLM":
            self.norm = FiLMLayer(num_features=ln_in_dim,
                                cond_dim=cond_channels,
                                activation="SiLU")
        elif norm_layer == "AdaLN":
            self.norm = AdaLNLayer(num_features=ln_in_dim,
                                    cond_dim=cond_channels,
                                    activation="SiLU")
        elif norm_layer == "LayerNorm":
            self.norm = nn.LayerNorm(ln_in_dim)
        else:
            raise ValueError(f"Unknown norm_layer: {norm_layer}")

        self.mixing_layer = nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.GLU(dim=-1),
            nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()
        )
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            if isinstance(self.norm, (FiLMLayer, AdaLNLayer)):
                y = self.norm(x, cond)
            else:
                y = self.norm(x)
            y = self.mixing_layer(y)
        else:
            y = self.mixing_layer(x)
            if isinstance(self.norm, (FiLMLayer, AdaLNLayer)):
                y = self.norm(y, cond)
            else:
                y = self.norm(y)
        if self.residual_learning:
            return (x + y)
        else:
            return y


class GatedLinearUnitFusionLayer(BaseFusionLayer):
    """
    Gated Linear Unit (GLU) based fusion layer that combines multiple input channels.
    
    This layer concatenates multiple input tensors and processes them through multiple
    FusionBlocks containing gated linear unit mechanisms. It supports optional 
    normalization (LayerNorm, AdaLN, or FiLM) that can be applied either before or 
    after the mixing layer.
    
    The GLU mechanism uses a linear layer that projects to 2x the output channels,
    then applies a gating function along the last dimension, effectively learning
    which features to pass through.
    
    Args:
        in_channels: List of input channel dimensions to be concatenated
        out_channels: Number of output channels after the final readout layer
        cond_dim: Dimension of conditioning vector for adaptive normalization layers
        dropout_p: Dropout probability applied after each GLU operation
        residual_learning: Whether to use residual connections in FusionBlocks
        norm_first: If True, apply normalization before the mixing layer; 
                   if False, apply after
        norm_layer: Type of normalization to use. Options are:
                   - "LayerNorm": Standard layer normalization
                   - "AdaLN": Adaptive Layer Normalization conditioned on cond vector
                   - "FiLM": Feature-wise Linear Modulation conditioned on cond vector
        n_blocks: Number of FusionBlocks to stack (default: 1)
        hidden_channels: Number of channels in intermediate FusionBlocks
    """

    def __init__(self, 
                 in_channels: list[int],
                 out_channels: int,
                 cond_dim: int = 0,
                 dropout_p: float = 0.0,
                 residual_learning: bool = True,
                 norm_first: bool = True,
                 norm_layer: str = "AdaLN",
                 n_blocks: int = 1,
                 hidden_channels: int = 128
                 ):
        
        super().__init__(in_channels, out_channels)

        self.norm_first = norm_first
        
        blocks = [FusionBlock(in_channels=sum(in_channels),
                              out_channels=hidden_channels,
                              cond_channels=cond_dim,
                              norm_first=norm_first,
                              norm_layer=norm_layer,
                              dropout_p=dropout_p)
                  ]
        
        
        
        for _ in range(n_blocks-1):
            blocks.append(FusionBlock(in_channels=hidden_channels,
                                      out_channels=hidden_channels,
                                      cond_channels=cond_dim,
                                      norm_first=norm_first,
                                      norm_layer=norm_layer,
                                      dropout_p=dropout_p))
        self.blocks = nn.ModuleList(blocks)
        self.readout = nn.Linear(hidden_channels, out_channels)

    def forward(self, *inputs: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that concatenates inputs and applies normalization and mixing operations.
        The processing order depends on the norm_first flag:
        - If norm_first is True: applies normalization first, then mixing layer
        - If norm_first is False: applies mixing layer first, then normalization
        For conditional normalization layers (FiLMLayer, AdaLNLayer), the conditioning
        tensor is passed to the normalization layer. For other normalization types,
        only the input tensor is used.
        Args:
            *inputs: Variable number of input tensors to be concatenated along the last dimension
            cond: Conditioning tensor used by conditional normalization layers
        Returns:
            Processed tensor after normalization and mixing operations
        """
        x = torch.cat(inputs, dim=-1)
        
        for block in self.blocks:
            x = block(x, cond)
        return self.readout(x)