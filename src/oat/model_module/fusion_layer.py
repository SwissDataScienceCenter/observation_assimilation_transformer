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
        hidden_dim: MLP hidden size (default: 32)
    """

    def __init__(self, 
                 *,
                 num_features: int, 
                 cond_dim: int, 
                 hidden_dim: int = 32,
                 activation: str = "SiLU"):
        super().__init__()

        act_fn = get_activation(activation)
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
    
    
class AdaLNLayer(FiLMLayer):
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
                 hidden_dim: int = 32,
                 activation: str = "SiLU",
                 eps: float = 1e-5):
        super().__init__(num_features=num_features, 
                         cond_dim=cond_dim, 
                         hidden_dim=hidden_dim,
                         activation=activation)
        
        self.eps = eps
        self.ln = nn.LayerNorm(num_features, eps=eps, elementwise_affine=False)
    

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
        
        return super().forward(x_norm, cond)


class BaseFusionLayer(nn.Module):
    """
    Base class for fusion layers that combine multiple input tensors into a single output tensor.
    Attributes:
        in_channels: List of input channel counts for each input tensor.
        out_channels: Number of output channels for the fused tensor.
    """

    def __init__(self, in_channels: list[int], out_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, *inputs: torch.Tensor, cond: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass


class GLUFusionBlock(nn.Module):
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
        if in_channels != out_channels and residual_learning:
            Warning("Residual learning requires in_channels to equal out_channels. "
                          "Will ignore residual_learning.")
            residual_learning = False
            
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
            if isinstance(self.norm, (FiLMLayer , AdaLNLayer)):
                y = self.norm(x, cond)
            else:
                y = self.norm(x)
            y = self.mixing_layer(y)
        else:
            y = self.mixing_layer(x)
            if isinstance(self.norm, (FiLMLayer , AdaLNLayer)):
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
        cond_dim: Conditioning dimension(s). Pass an int for a single conditioning
                  tensor, or a list[int] when passing multiple conditioning tensors
                  (they will be concatenated; the list encodes their individual sizes).
        dropout_p: Dropout probability applied after each GLU operation
        residual_learning: Whether to use residual connections in FusionBlocks. First block won't use it due to dimension mismatch.
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
                 cond_dim: int | list[int] = 0,
                 dropout_p: float = 0.0,
                 residual_learning: bool = True,
                 norm_first: bool = True,
                 norm_layer: str = "AdaLN",
                 n_blocks: int = 1,
                 hidden_channels: int = 128
                 ):

        super().__init__(in_channels, out_channels)

        total_cond_dim = sum(cond_dim) if isinstance(cond_dim, list) else cond_dim

        blocks = [GLUFusionBlock(in_channels=sum(in_channels),
                              out_channels=hidden_channels,
                              cond_channels=total_cond_dim,
                              norm_first=norm_first,
                              norm_layer=norm_layer,
                              dropout_p=dropout_p)
                  ]

        for _ in range(n_blocks-1):
            blocks.append(GLUFusionBlock(in_channels=hidden_channels,
                                      out_channels=hidden_channels,
                                      cond_channels=total_cond_dim,
                                      norm_first=norm_first,
                                      norm_layer=norm_layer,
                                      dropout_p=dropout_p,
                                      residual_learning=residual_learning))
        self.blocks = nn.ModuleList(blocks)
        self.readout = nn.Linear(hidden_channels, out_channels)

    def forward(self, *inputs: torch.Tensor, cond: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
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
            cond: Single conditioning tensor, or a tuple of tensors concatenated along the last
                  dimension. When a tuple is passed, individual sizes must match cond_dim at
                  construction time.
        Returns:
            Processed tensor after normalization and mixing operations
        """
        x = torch.cat(inputs, dim=-1)
        if isinstance(cond, tuple):
            cond = torch.cat(cond, dim=-1)

        for block in self.blocks:
            x = block(x, cond)
        return self.readout(x)


class LearnableGating(nn.Module):
    
    def __init__(self, in_channels: int, 
                 out_channels: int, 
                 hidden_channels: int = 32,
                 n_layers: int = 2,
                 activation: str = "GELU"):
        super().__init__()
        
        layers = []
        in_size = in_channels
        out_size = hidden_channels
        
        for i in range(n_layers - 1):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(get_activation(activation))
            in_size = out_size
            
        layers.append(nn.Linear(in_size, out_channels)) # TODO should this layer have a bias?
        layers.append(nn.Sigmoid())
        
        self.ffnn = nn.Sequential(*layers)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffnn(x)

class ObservationEmbedding(nn.Module):
    """Embed each observation variable, with a learned token for missing values.

    Each variable v has its own MLP ``1 -> hidden_channels -> d_embed``. Where the
    mask is 0, the layer uses the learned vector ``missing_token[v]`` instead of
    the MLP output. Thus a missing value is never the same as an observed value,
    and the value at a masked entry has no effect (it can also be NaN).

    Args:
        n_vars: Number of observation variables V.
        d_embed: Embedding size for each variable.
        hidden_channels: Hidden size of each per-variable MLP.
        activation: Activation name for ``get_activation``.

    Shapes:
        obs, mask: [..., V]. ``mask > 0.5`` means observed.
        output: [..., V * d_embed], variables in input order.
    """

    def __init__(
        self,
        n_vars: int,
        d_embed: int = 8,
        hidden_channels: int = 16,
        activation: str = "GELU",
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.d_embed = d_embed
        self.out_channels = n_vars * d_embed
        lin1 = [nn.Linear(1, hidden_channels) for _ in range(n_vars)]
        lin2 = [nn.Linear(hidden_channels, d_embed) for _ in range(n_vars)]
        self.w1 = nn.Parameter(torch.stack([m.weight[:, 0] for m in lin1]))
        self.b1 = nn.Parameter(torch.stack([m.bias for m in lin1]))
        self.w2 = nn.Parameter(torch.stack([m.weight.T for m in lin2]))
        self.b2 = nn.Parameter(torch.stack([m.bias for m in lin2]))
        self.act = get_activation(activation)
        self.missing_token = nn.Parameter(torch.randn(n_vars, d_embed) * 0.02)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        observed = (mask > 0.5).unsqueeze(-1)  # [..., V, 1]
        # Set masked inputs to 0 before the MLP, so NaN values give no NaN gradient.
        obs_v = obs.unsqueeze(-1)
        y = torch.where(observed, obs_v, torch.zeros_like(obs_v))
        h = self.act(y * self.w1 + self.b1)  # [..., V, H]
        e = torch.einsum("...vh,vhd->...vd", h, self.w2) + self.b2  # [..., V, D]
        e = torch.where(observed, e, self.missing_token)
        return e.flatten(-2)
