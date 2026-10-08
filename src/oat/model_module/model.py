"""
Observation Assimilation Transformer (OAT) for Neural Downscaling.

Following architecture from Neural Downscaling with Station Observation.
"""

from typing import Dict, Tuple, Optional, Any, Sequence
import importlib
import torch
from torch import nn
from dataclasses import dataclass
from omegaconf import DictConfig

from oat.model_module import (
    global_encoder,
    topographic_encoder,
    fusion_layer,
    output_head,
    transformer,
    masking,
)


def _search_coords(group: dict[str, torch.Tensor]) -> torch.Tensor:
    """Coords for neighbour search: projected ``search_coords`` if present,
    else raw ``coords`` (unchanged behaviour for non-projecting datasets)."""
    return group.get("search_coords", group["coords"])


@dataclass
class GlobalEncoderConfig:
    """Configuration for global encoder (Eq. 1)."""

    in_channels: int  # data dependent
    field_extent_min: Tuple[float, float]  # = (2255500., 840500.)    # data dependent
    field_extent_max: Tuple[float, float]  # = (2964500., 1479500.)   # data dependent
    type: str = "UNet"
    out_channels: int = 32
    spatial_dims: int = 2
    patch_size: int = 2
    window_size: int = 7
    channels: Tuple[int, ...] = (16, 32, 64, 128)
    strides: Tuple[int, ...] = (1, 1, 1)
    num_res_units: int = 2
    kernel_size: int = 3
    num_blocks: int = 4
    hidden_dims: Tuple[int, ...] = (32,)
    act_fn: str = "GELU"
    use_layernorm: bool = False
    use_batchnorm: bool = False


@dataclass
class TopographicEncoderConfig:
    """Configuration for topographic encoder (Eq. 4-5)."""

    in_channels: int  # data dependent
    out_channels: int = 64
    n_layers: int = 3
    initial_hw: Tuple[int, int] = (32, 32)
    kernel_sizes: int = 3
    act_fn: str = "GELU"
    use_batchnorm: bool = True
    type: str = "CNN"
    hidden_dims: tuple[int, ...] = ()
    use_layernorm: bool = False


@dataclass
class NwpTopoFusionConfig:
    """Configuration for the NWP+topo fusion layer (Eq. 6-7), used by both UninformedOAT and OAT."""

    cond_dim: int  # time embedding dimension — data-dependent, must be set explicitly
    out_channels: int = 64
    hidden_channels: int = 128
    dropout_p: float = 0.1
    norm_layer: str = "AdaLN"
    norm_first: bool = True
    n_blocks: int = 1
    residual_learning: bool = True


@dataclass
class InnovationFusionConfig:
    """Configuration for the innovation fusion layer (Eq. 16), used only by OAT.

    When condition_on_mask=True, cond_dim must be [time_dim, mask_dim] and the
    observation mask is concatenated to the time embedding as a second conditioning
    signal. When False, cond_dim is a plain int (time only).
    Both are data-dependent and must be set explicitly in the training script.
    """

    cond_dim: int | list[int]  # data-dependent, must be set explicitly
    condition_on_mask: bool = False
    out_channels: int = 64
    hidden_channels: int = 128
    dropout_p: float = 0.1
    norm_layer: str = "AdaLN"
    norm_first: bool = True
    n_blocks: int = 1
    residual_learning: bool = True


@dataclass
class LearnableGatingConfig:
    """Configuration for the learnable gating layer."""

    in_channels: int  # data-dependent, must be set explicitly
    out_channels: int  # data-dependent, must be set explicitly
    hidden_channels: int = 32
    n_layers: int = 2
    activation: str = "GELU"


@dataclass
class TransformerConfig:
    """Configuration for OAT transformer block (Eq. 17-25)."""

    d_geo_bias: int
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    activation: str = "GELU"
    sparse_attn: bool = True
    decoder_self_attention: bool = False
    dim_geo_bias_feedforward: int = 16
    activation_geo_bias: str = "GELU"
    geo_attn_mode: str = "additive"


@dataclass
class MaskingConfig:
    """Configuration for spatial neighborhood search (Eq. 14)."""

    mode: str = "knn"
    radius: float = 1.0
    k: int = 4
    max_num_neighbors: int = 32


@dataclass
class OutputHeadConfig:
    """Configuration for output head (Eq. 26)."""

    in_features: int
    variable_to_distribution: Dict[str, str]
    loss_scaling: Sequence[float] | torch.Tensor | None = None
    crps_mc_samples: int = 50
    observation_assimilation_as: str = "innovation"


@dataclass
class OATConfig:
    """Complete OAT model configuration."""

    global_encoder: GlobalEncoderConfig
    topographic_encoder: TopographicEncoderConfig
    nwp_topo_fusion: NwpTopoFusionConfig
    innovation_fusion: InnovationFusionConfig
    transformer: TransformerConfig
    masking: MaskingConfig
    output_head: OutputHeadConfig
    learnable_gating: LearnableGatingConfig | None = None
    num_samples: int = 100
    pre_oat_loss_weight: float = 0.0
    mask_innovations: bool = True

    @classmethod
    def default(cls) -> "OATConfig":
        """Create default configuration."""
        d_model = 64
        time_dim = 4
        n_vars = 1
        return cls(
            global_encoder=GlobalEncoderConfig(out_channels=d_model // 2),
            topographic_encoder=TopographicEncoderConfig(out_channels=d_model // 2),
            nwp_topo_fusion=NwpTopoFusionConfig(
                cond_dim=time_dim, hidden_channels=d_model, out_channels=d_model
            ),
            innovation_fusion=InnovationFusionConfig(
                cond_dim=time_dim, hidden_channels=d_model, out_channels=d_model
            ),
            transformer=TransformerConfig(d_model=d_model, dim_feedforward=2 * d_model),
            masking=MaskingConfig(),
            output_head=OutputHeadConfig(in_features=d_model),
            learnable_gating=LearnableGatingConfig(
                in_channels=n_vars, hidden_channels=d_model // 2, out_channels=d_model
            ),
            num_samples=100,
        )


@dataclass
class UninformedOATConfig:
    global_encoder: GlobalEncoderConfig
    topographic_encoder: TopographicEncoderConfig
    nwp_topo_fusion: NwpTopoFusionConfig
    # None builds only the Eq. 1-7 trunk (no output head). Used when another
    # model consumes the fusion tokens and supplies its own likelihood, e.g.
    # GPBaseline using this stack as its mean function.
    output_head: Optional[OutputHeadConfig] = None
    num_samples: int = 100

    @classmethod
    def default(cls) -> "UninformedOATConfig":
        """Create default configuration."""
        d_model = 64
        return cls(
            global_encoder=GlobalEncoderConfig(out_channels=d_model // 2),
            topographic_encoder=TopographicEncoderConfig(out_channels=d_model // 2),
            nwp_topo_fusion=NwpTopoFusionConfig(
                cond_dim=4, hidden_channels=d_model, out_channels=d_model
            ),
            output_head=OutputHeadConfig(in_features=d_model),
            num_samples=100,
        )


def build_topographic_encoder(
    tc: TopographicEncoderConfig,
) -> topographic_encoder.TopographyEncoder:
    """Dispatch to CNN (grid patches) or MLP (flat field values) by ``tc.type``."""
    if getattr(tc, "type", "CNN") == "CNN":
        return topographic_encoder.CNNTopographyEncoder(
            in_channels=tc.in_channels,
            out_channels=tc.out_channels,
            n_layers=tc.n_layers,
            initial_hw=tc.initial_hw,
            kernel_sizes=tc.kernel_sizes,
            act_fn=tc.act_fn,
            use_batchnorm=tc.use_batchnorm,
        )
    if tc.type == "MLP":
        return topographic_encoder.MLPTopographyEncoder(
            in_channels=tc.in_channels,
            out_channels=tc.out_channels,
            hidden_dims=tc.hidden_dims,
            act_fn=tc.act_fn,
            use_layernorm=tc.use_layernorm,
            use_batchnorm=tc.use_batchnorm,
        )
    raise ValueError(f"Unsupported topographic encoder type: {tc.type}")


class UninformedOAT(nn.Module):
    """
    Base class for Observation Assimilation Transformer for neural downscaling.
    This class implements the "uninformed" computations for OAT, i.e it does not
    implement any conditioning on station measurements.
    This model can be used as a baseline that does not assimilate observations that
    learns to correct pointwise corrections of NWP systematic errors
    (known as statistical postprocessing).
    """

    requires_observation: bool = False

    def __init__(self, config: Optional[UninformedOATConfig] = None) -> None:
        super().__init__()

        if config is None:
            config = UninformedOATConfig.default()
        self.config = config

        d_global = config.global_encoder.out_channels
        d_topo = config.topographic_encoder.out_channels

        self.global_encoder = self._init_global_encoder(config=config)
        self.topographic_encoder = self._init_topographic_encoder(config=config)
        self.fusion_layer1 = fusion_layer.GatedLinearUnitFusionLayer(
            in_channels=[d_global, d_topo],
            out_channels=config.nwp_topo_fusion.out_channels,
            hidden_channels=config.nwp_topo_fusion.hidden_channels,
            cond_dim=config.nwp_topo_fusion.cond_dim,
            dropout_p=config.nwp_topo_fusion.dropout_p,
            norm_layer=config.nwp_topo_fusion.norm_layer,
            norm_first=config.nwp_topo_fusion.norm_first,
            n_blocks=config.nwp_topo_fusion.n_blocks,
            residual_learning=config.nwp_topo_fusion.residual_learning,
        )
        self.output_head = (
            self._init_output_head(config=config)
            if config.output_head is not None
            else None
        )

        self.num_samples = config.num_samples

    # Returns Any, like the untyped _init_output_head it mirrors: OutputHead's
    # own annotations don't match its runtime returns yet, so a precise type
    # here surfaces pre-existing errors in every caller.
    def _require_output_head(self) -> Any:
        if self.output_head is None:
            raise RuntimeError(
                "This model was built without an output head "
                "(config.output_head is None); only _create_uninformed_tokens "
                "is available."
            )
        return self.output_head

    def _init_global_encoder(self, config):
        if config.global_encoder.type == "UNet":
            return global_encoder.UNetGlobalEncoder(
                unet_params={
                    "in_channels": config.global_encoder.in_channels,
                    "out_channels": config.global_encoder.out_channels,
                    "spatial_dims": config.global_encoder.spatial_dims,
                    "channels": config.global_encoder.channels,
                    "strides": config.global_encoder.strides,
                    "num_res_units": config.global_encoder.num_res_units,
                },
                field_extent={
                    "min": torch.tensor(config.global_encoder.field_extent_min),
                    "max": torch.tensor(config.global_encoder.field_extent_max),
                },
            )
        elif config.global_encoder.type == "SwinUNETR":
            return global_encoder.SwinUNETRGlobalEncoder(
                swin_unetr_params={
                    "in_channels": config.global_encoder.in_channels,
                    "out_channels": config.global_encoder.out_channels,
                    "spatial_dims": config.global_encoder.spatial_dims,
                    "patch_size": config.global_encoder.patch_size,
                    "window_size": config.global_encoder.window_size,
                },
                field_extent={
                    "min": torch.tensor(config.global_encoder.field_extent_min),
                    "max": torch.tensor(config.global_encoder.field_extent_max),
                },
            )
        elif config.global_encoder.type == "MLP":
            return global_encoder.MLPGlobalEncoder(
                mlp_params={
                    "in_channels": config.global_encoder.in_channels,
                    "out_channels": config.global_encoder.out_channels,
                    "hidden_dims": config.global_encoder.hidden_dims,
                    "act_fn": config.global_encoder.act_fn,
                },
                field_extent={
                    "min": torch.tensor(config.global_encoder.field_extent_min),
                    "max": torch.tensor(config.global_encoder.field_extent_max),
                },
            )
        elif config.global_encoder.type == "ResNetCNN":
            self.global_encoder = global_encoder.ResNetCNNGlobalEncoder(
                resnet_cnn_params={
                    "in_channels": config.global_encoder.in_channels,
                    "out_channels": config.global_encoder.out_channels,
                    "kernel_size": config.global_encoder.kernel_size,
                    "num_blocks": config.global_encoder.num_blocks,
                    "act_fn": config.global_encoder.act_fn,
                },
                field_extent={
                    "min": torch.tensor(config.global_encoder.field_extent_min),
                    "max": torch.tensor(config.global_encoder.field_extent_max),
                },
            )
        elif config.global_encoder.type == "Point":
            return global_encoder.PointGlobalEncoder(
                mlp_params={
                    "in_channels": config.global_encoder.in_channels,
                    "out_channels": config.global_encoder.out_channels,
                    "hidden_dims": config.global_encoder.hidden_dims,
                    "act_fn": config.global_encoder.act_fn,
                    "use_layernorm": config.global_encoder.use_layernorm,
                    "use_batchnorm": config.global_encoder.use_batchnorm,
                },
            )

        raise ValueError(
            f"Unsupported global encoder type: {config.global_encoder.type}. Supported types are 'UNet', 'SwinUNETR', 'MLP', 'ResNetCNN', and 'Point'."
        )

    def _init_topographic_encoder(self, config):
        return build_topographic_encoder(config.topographic_encoder)

    def _init_output_head(self, config):

        return output_head.OutputHead(
            in_features=config.output_head.in_features,
            variable_to_distribution=config.output_head.variable_to_distribution,
            loss_scaling=config.output_head.loss_scaling,
            crps_mc_samples=config.output_head.crps_mc_samples,
            observation_assimilation_as=config.output_head.observation_assimilation_as,
        )

    def forward(
        self,
        global_data: Dict[str, torch.Tensor],
        query_data: Dict[str, torch.Tensor],
        num_samples: int | None = None,
        differentiable_samples: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            global_data: {
                'nwp': (B, C, H_lr, W_lr) - NWP fields
                'time_emb': (B, D_time) - time embeddings
            }
            query_data: {
                'coords': (B, N_q, 2) - query coordinates
                'topo': (B, N_q, C_topo, H_hr, W_hr) - topographic patches
                'geo': (B, N_q, D) - query coordinates
            }
            num_samples: Number of samples to draw from output distribution. If None, uses default from config.
            differentiable_samples: Whether to sample with a reparametrization

        Returns:
            Predictions at query locations: (B, N_q, C_y) or dict with distribution params
        """

        head = self._require_output_head()
        query_token = self._create_uninformed_tokens(
            global_data=global_data, local_data=query_data
        )
        out = head(query_token)
        with torch.no_grad():
            if num_samples is None:
                num_samples = self.num_samples
            samples = head.sample(
                x=None,
                pred_dist=out,
                num_samples=num_samples,
                differentiable=differentiable_samples,
            )
        out["samples"] = samples

        return out

    def forward_with_loss(
        self,
        global_data: Dict[str, torch.Tensor],
        query_data: Dict[str, torch.Tensor],
        target_obs: torch.Tensor,
        target_mask_obs: torch.Tensor,
        num_samples: int | None = None,
        differentiable_samples: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            global_data: {
                'nwp': (B, C, H_lr, W_lr) - NWP fields
                'time_emb': (B, D_time) - time embeddings
            }
            query_data: {
                'coords': (B, N_q, 2) - query coordinates
                'topo': (B, N_q, C_topo, H_hr, W_hr) - topographic patches
                'obs': (B, N_q, C_y) - ground truth observations
                'mask_obs': (B, N_q, C_y) - observation mask
                'geo': (B, N_q, D) - query coordinates
            }
            num_samples: Number of samples to draw from output distribution. If None, uses default from config.
            differentiable_samples: Whether to sample with a reparametrization

        Returns:
            predictions: Dict with model outputs
            per_var_losses: Dict mapping variable name to its detached scaled loss scalar
            loss: Scalar loss tensor (sum of per-variable scaled losses)
        """

        head = self._require_output_head()
        query_token = self._create_uninformed_tokens(
            global_data=global_data, local_data=query_data
        )
        out, per_var_losses, loss = head.loss(query_token, target_obs, target_mask_obs)

        with torch.no_grad():
            if num_samples is None:
                num_samples = self.num_samples
            samples = head.sample(
                x=None,
                pred_dist=out,
                num_samples=num_samples,
                differentiable=differentiable_samples,
            )

        out["samples"] = samples

        return out, per_var_losses, loss

    def _create_uninformed_tokens(
        self, global_data: Dict[str, torch.Tensor], local_data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Create uninformed tokens from global and local information.

        Implements Eq. 1-7:
        1. Z_nwp = G(U_nwp, τ)  - Global encoding
        2. Z_nwp_x = R(Z_nwp, x)  - Spatial readout
        3. Z_ξ = E_ξ(ξ)  - Topographic encoding
        4. Z = F([Z_nwp || Z_ξ], τ)  - Fusion

        Args:
            global_data: Global NWP fields and time embeddings
            local_data: Local coordinates and topographic patches

        Returns:
            Uninformed tokens: (B, N, D_model)
        """
        # Global encoding and readout (Eq. 1-3)
        global_token = self.global_encoder(global_data, local_data)

        # Topographic encoding (Eq. 4-5)
        topo_token = self.topographic_encoder(local_data["topo"])

        # Fusion (Eq. 6-7)
        batch_size = global_token.shape[0]
        n_points = global_token.shape[1]
        cond = global_data["time_emb"].unsqueeze(1).expand(batch_size, n_points, -1)
        uninformed_token = self.fusion_layer1(global_token, topo_token, cond=cond)

        return uninformed_token


class OAT(UninformedOAT):
    """
    Observation Assimilation Transformer for neural downscaling.

    Architecture:
    1. Global encoder: Znwp = G(Unwp, τ)  [Eq. 1]
    2. Local readout: Zq, Zkv = R(Znwp, x)  [Eq. 2-3]
    3. Topographic encoder: Zξ = Eξ(ξ)  [Eq. 4-5]
    4. Fusion: Z = F([Znwp || Zξ], τ)  [Eq. 6-7]
    5. Innovation: δy = H(Z, y)  [Eq. 8-12]
    6. Station tokens: Skv = Fusion(Z, δy, τ)  [Eq. 16 - initial encoding]
    7. Station encoding + Attention: [Eq. 16-25]
       - Encoder self-attention refines Skv (this IS Encstation from Eq. 16)
       - Decoder cross-attention: Z̃ = Attn(Q=Zq, K=V=Skv)
    8. Output: ŷ = Hout(Z̃, τ)  [Eq. 26]

    Note: The transformer encoder layers implement the station encoding from Eq. 16.
    The fusion layer creates initial station tokens, then encoder self-attention
    refines them by aggregating information across stations.
    """

    requires_observation: bool = True
    # OATConfig always carries an output head, unlike UninformedOATConfig where
    # it is optional; narrow it so the uses below need no None checks. Any
    # rather than OutputHead, for the reason given on _require_output_head.
    output_head: Any

    def __init__(self, config: Optional[OATConfig] = None) -> None:
        """
        Initialize OAT model.

        Args:
            config: Model configuration. If None, uses default config.
        """

        super().__init__(
            config=UninformedOATConfig(
                global_encoder=config.global_encoder,
                topographic_encoder=config.topographic_encoder,
                nwp_topo_fusion=config.nwp_topo_fusion,
                output_head=config.output_head,
                num_samples=config.num_samples,
            )
        )

        if config is None:
            config = OATConfig.default()
        self.config = config
        self.mask_innovations = config.mask_innovations
        self.register_buffer(
            "_pre_oat_loss_weight",
            torch.tensor(self.config.pre_oat_loss_weight, dtype=torch.float32),
        )

        # Extract dimensions
        d_model = config.transformer.d_model

        # Fusion layer 2: Uninformed token + Innovation → Initial station token
        # Note: Full station encoding (Eq. 16) completed by transformer encoder
        self.fusion_layer2 = fusion_layer.GatedLinearUnitFusionLayer(
            in_channels=[d_model, self.output_head.target_features],
            out_channels=config.innovation_fusion.out_channels,
            hidden_channels=config.innovation_fusion.hidden_channels,
            cond_dim=config.innovation_fusion.cond_dim,
            dropout_p=config.innovation_fusion.dropout_p,
            norm_layer=config.innovation_fusion.norm_layer,
            norm_first=config.innovation_fusion.norm_first,
            n_blocks=config.innovation_fusion.n_blocks,
            residual_learning=config.innovation_fusion.residual_learning,
        )

        # Spatial neighborhood search (Eq. 14)
        self.mask_generator = masking.SpatialNeighborSearch(
            mode=config.masking.mode,
            radius=config.masking.radius,
            k=config.masking.k,
            max_num_neighbors=config.masking.max_num_neighbors,
        )

        # Observation Assimilation Transformer (Eq. 17-25)
        self.oat_block = transformer.ObservationAssimilationTransformer(
            d_model=config.transformer.d_model,
            nhead=config.transformer.nhead,
            num_encoder_layers=config.transformer.num_encoder_layers,
            num_decoder_layers=config.transformer.num_decoder_layers,
            dim_feedforward=config.transformer.dim_feedforward,
            dropout=config.transformer.dropout,
            activation=config.transformer.activation,
            sparse_attn=config.transformer.sparse_attn,
            d_geo_bias=config.transformer.d_geo_bias,
            decoder_self_attention=config.transformer.decoder_self_attention,
            dim_geo_bias_feedforward=config.transformer.dim_geo_bias_feedforward,
            activation_geo_bias=config.transformer.activation_geo_bias,
            geo_attn_mode=config.transformer.geo_attn_mode,
        )

        if config.learnable_gating is not None:
            self.mask_gating = fusion_layer.LearnableGating(
                in_channels=self.output_head.target_features,
                out_channels=d_model,
                hidden_channels=config.learnable_gating.hidden_channels,
                n_layers=config.learnable_gating.n_layers,
                activation=config.learnable_gating.activation,
            )

    @property
    def pre_oat_loss_weight(self) -> float:
        return float(self._pre_oat_loss_weight.item())

    @pre_oat_loss_weight.setter
    def pre_oat_loss_weight(self, value: float) -> None:
        self._pre_oat_loss_weight.fill_(float(value))

    def forward(
        self,
        global_data: Dict[str, torch.Tensor],
        observation_data: Dict[str, torch.Tensor],
        query_data: Dict[str, torch.Tensor],
        num_samples: int | None = None,
        differentiable_samples: bool = False,
        return_pre_oat_output: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass through OAT model.

        Args:
            global_data: {
                'nwp': (B, C, H_lr, W_lr) - NWP fields
                'time_emb': (B, D_time) - time embeddings
            }
            observation_data: {
                'coords': (B, N_kv, 2) - station coordinates
                'topo': (B, N_kv, C_topo, H_hr, W_hr) - topographic patches
                'geo': (B, N_kv, D) - station coordinates
                'obs': (B, N_kv, C_y) - observations
                'mask_obs': (B, N_kv, C_y) - observation mask
            }
            query_data: {
                'coords': (B, N_q, 2) - query coordinates
                'topo': (B, N_q, C_topo, H_hr, W_hr) - topographic patches
                'geo': (B, N_q, D) - query coordinates
            }
            num_samples: Number of samples to draw from output distribution. If None, uses default from config.
            differentiable_samples: Whether to sample with a reparametrization
            return_pre_oat_output: If True, also returns output before final output head for analysis.

        Returns:
            Predictions at query locations: (B, N_q, C_y) or dict with distribution params
        """

        oat_out, pre_oat_out = self._forward(global_data, observation_data, query_data)
        out = self.output_head(oat_out)

        with torch.no_grad():
            if num_samples is None:
                num_samples = self.num_samples
            samples = self.output_head.sample(
                x=None,
                pred_dist=out,
                num_samples=num_samples,
                differentiable=differentiable_samples,
            )
        out["samples"] = samples

        if return_pre_oat_output:
            pre_oat_output = self.output_head(pre_oat_out)
            if num_samples is None:
                num_samples = self.num_samples
            with torch.no_grad():
                pre_oat_samples = self.output_head.sample(
                    x=None,
                    pred_dist=pre_oat_output,
                    num_samples=num_samples,
                    differentiable=differentiable_samples,
                )
            pre_oat_output["samples"] = pre_oat_samples
            out["pre_oat_output"] = pre_oat_output

        return out

    def forward_with_loss(
        self,
        global_data: Dict[str, torch.Tensor],
        observation_data: Dict[str, torch.Tensor],
        query_data: Dict[str, torch.Tensor],
        target_obs: torch.Tensor,
        target_mask_obs: torch.Tensor,
        num_samples: int | None = None,
        differentiable_samples: bool = False,
        return_pre_oat_output: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with loss computation.

        Args:
            Same as forward(), but query_data must also contain:
                'obs': (B, N_q, C_y) - ground truth observations
                'geo': (B, N_q, D) - query coordinates
                'mask_obs': (B, N_q, C_y) - observation mask

        Returns:
            predictions: Dict with model outputs
            per_var_losses: Dict mapping variable name to its detached scaled loss scalar (blended with pre-OAT when pre_oat_loss_weight > 0, so values always sum to loss)
            loss: Scalar loss tensor (sum of per-variable scaled losses, blended with pre-OAT loss if applicable)
        """

        oat_out, pre_oat_out = self._forward(global_data, observation_data, query_data)
        out, per_var_losses, loss = self.output_head.loss(
            oat_out, target_obs, target_mask_obs
        )
        if self.pre_oat_loss_weight > 0:
            _, pre_per_var_losses, pre_loss = self.output_head.loss(
                pre_oat_out, target_obs, target_mask_obs
            )
            pre_weight = max(0.0, min(1.0, float(self.pre_oat_loss_weight)))
            loss = (1 - pre_weight) * loss + pre_weight * pre_loss
            # Scale the per-var losses by the pre-oat weight as done with the overall loss scalar
            per_var_losses = {
                var: (1 - pre_weight) * v + pre_weight * pre_per_var_losses[var]
                for var, v in per_var_losses.items()
            }

        with torch.no_grad():
            if num_samples is None:
                num_samples = self.num_samples
            samples = self.output_head.sample(
                x=None,
                pred_dist=out,
                num_samples=num_samples,
                differentiable=differentiable_samples,
            )

        if return_pre_oat_output:
            with torch.no_grad():
                pre_oat_output = self.output_head(pre_oat_out)
                if num_samples is None:
                    num_samples = self.num_samples
                pre_oat_samples = self.output_head.sample(
                    x=None,
                    pred_dist=pre_oat_output,
                    num_samples=num_samples,
                    differentiable=differentiable_samples,
                )
            pre_oat_output["samples"] = pre_oat_samples
            out["pre_oat_output"] = pre_oat_output

        out["samples"] = samples

        return out, per_var_losses, loss

    def _forward(
        self,
        global_data: Dict[str, torch.Tensor],
        observation_data: Dict[str, torch.Tensor],
        query_data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core forward pass implementing the architecture.

        Steps:
        1. Create uninformed tokens for stations (Eq. 1-7)
        2. Compute innovation at stations (Eq. 8-12)
        3. Create initial station tokens with innovation (Eq. 16 - fusion part)
        4. Create uninformed tokens for queries (Eq. 1-7)
        5. Apply transformer (Eq. 16-25):
           - Encoder: Self-attention refines station tokens (Eq. 16 - Encstation)
           - Decoder: Cross-attention from queries to stations (Eq. 17-25)

        Returns:
            Updated query tokens: (B, N_q, D_model)
        """
        # Step 1: Create uninformed station tokens (Eq. 1-7)
        uninformed_kv_token = self._create_uninformed_tokens(
            global_data=global_data, local_data=observation_data
        )

        # Step 2: Compute innovation at stations (Eq. 8-12)
        innovation = self.output_head.compute_innovation(
            uninformed_kv_token,
            observation_data["obs"],
            mask=observation_data["mask_obs"] if self.mask_innovations else None,
        )

        # Step 3: Create initial station tokens with innovation (Eq. 16 - fusion)
        # Note: Full station encoding happens in transformer encoder via self-attention
        batch_size = uninformed_kv_token.shape[0]
        n_kv = uninformed_kv_token.shape[1]
        time_cond = global_data["time_emb"].unsqueeze(1).expand(batch_size, n_kv, -1)
        if self.config.innovation_fusion.condition_on_mask:
            cond = (time_cond, observation_data["mask_obs"])
        else:
            cond = time_cond

        station_token = self.fusion_layer2(uninformed_kv_token, innovation, cond=cond)

        # Apply learnable gating to station tokens based on observation mask
        # to allow the model to learn how much to trust the innovation signal at each station.
        if hasattr(self, "mask_gating"):
            mask_sigma = self.mask_gating(observation_data["mask_obs"])
            station_token = (
                1.0 - mask_sigma
            ) * uninformed_kv_token + mask_sigma * station_token

        # Step 4: Create uninformed query tokens (Eq. 1-7)
        query_token = self._create_uninformed_tokens(
            global_data=global_data, local_data=query_data
        )

        # Step 5: Create spatial masks (Eq. 14)
        if self.config.masking.mode == "full":
            # Full attention: every point attends to every other one, so there
            # is no neighbour search and no mask. Without edges, the attention
            # runs dense and unmasked.
            edges_src, edges_tgt, edges_memory = None, None, None
        else:
            edges_src = self.mask_generator(coords_1=_search_coords(observation_data))
            # Query-query edges are only read by decoder self-attention, so skip
            # the neighbour search when it is disabled.
            if self.config.transformer.decoder_self_attention:
                edges_tgt = self.mask_generator(coords_1=_search_coords(query_data))
            else:
                edges_tgt = None
            edges_memory = self.mask_generator(
                coords_1=_search_coords(observation_data),
                coords_2=_search_coords(query_data),
            )

        # Step 6: Apply transformer (Eq. 16-25)
        # - Encoder: Self-attention across stations refines station tokens (Eq. 16)
        # - Decoder: Cross-attention from queries to refined stations (Eq. 17-25)
        # Q = W_Q Z_q, K = W_K S_kv, V = W_V S_kv

        updated_query_token = self.oat_block(
            src=station_token,  # K, V from station tokens (Eq. 18-19)
            tgt=query_token,  # Q from query tokens (Eq. 17)
            src_mask_edges=edges_src,
            tgt_mask_edges=edges_tgt,
            memory_mask_edges=edges_memory,
            g_src=observation_data["geo"],  # Geometric bias for stations
            g_tgt=query_data["geo"],  # Geometric bias for queries
        )
        return updated_query_token, query_token


def get_model_cls(cfg: DictConfig) -> Any:
    model_module_str, model_cls_str = cfg.model.model_cls.rsplit(".", 1)
    model_module = importlib.import_module(model_module_str)
    return getattr(model_module, model_cls_str)
