from torch import nn
from oat.model_module import global_encoder, topographic_encoder, fusion_layer, output_head, transformer, masking
from typing import Any, Dict
import torch

class OAT(nn.Module):
    
    def __init__(self, config: Any) -> None:
        
        d_model = 64
        dropout_p = 0.1
        target_features = 1
        
        self.global_encoder = global_encoder.SwinUNETRGlobalEncoder(
            swin_unetr_params=config.model.global_encoder.swin_unetr_params,
            field_extent=config.data.field_extent
        )
        self.topographic_encoder = topographic_encoder.CNNTopographyEncoder(
            **config.model.topographic_encoder
        )
        
        in_channels = (config.model.global_encoder.swin_unetr_params.embed_dim +
                       config.model.topographic_encoder.out_channels)
        
        fusion_config = {
            "out_channels": d_model,
            "dropout_p": dropout_p,
            "norm_type": "AdaLN",
            "hidden_channels": d_model * 2
        }
        
        self.fusion_layer1 = fusion_layer.GatedLinearUnitFusionLayer(
            in_channels=in_channels,
            **fusion_config
        )
        
        output_head_dict = {
            "in_features": d_model,
            "target_features": target_features,
            "loss_fn": 'huber',
            "std_fn": 'softplus',
            "min_std_val": 1e-3,
            "degree": 20,
            "num_train_taus": 10
        }
        
        self.output_head = output_head.LinearOutputHead(
            in_features=output_head_dict["in_features"],
            target_features=output_head_dict["target_features"],
            loss_fn=output_head_dict["loss_fn"],
        )
        
        in_channels = [
            d_model,
            output_head_dict["target_features"]
        ]
        
        self.fusion_layer2 = fusion_layer.GatedLinearUnitFusionLayer(
            in_channels=in_channels,
            **fusion_config
        )
        
        mask_dict = {
            "mode": "knn",
            "radius": 1,
            "k": 4,
            "max_num_neigbours": 32
        }
        
        self.mask_generator = masking.SpatialNeighborSearch(
            **mask_dict
        )
        
        oat_dict = {
            "d_model": d_model,
            "nhead": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": d_model * 4,
            "dropout": dropout_p,
            "activation": "GELU",
            "sparse_attention": True,
            "d_geo_bias": 2,
            "dim_geo_bias_feedforward": 16,
        }
        self.oat_block = transformer.ObservationAssimilationTransformer(
            **oat_dict
        )
        
    def forward(self,
                global_data: Dict[str, torch.Tensor],
                observation_data: Dict[str, torch.Tensor],
                query_data: Dict[str, torch.Tensor]
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        
        # create obs tokens
        uninformed_obs_token = self._create_uninformed_tokens(
            global_data=global_data,
            local_data=observation_data
        )
        innovation = self.output_head.compute_innovation(
            uninformed_obs_token,
            observation_data['obs'],
            observation_data['mask_obs']
        )
        obs_token = self.fusion_layer2(
            uninformed_obs_token,
            innovation,
            cond=global_data['time_emb']
        )
        
        # create query token
        query_token = self._create_uninformed_tokens(
            global_data=global_data,
            local_data=query_data
        )
        
        # create mask
        mask_obs = self.mask_generator(
            coords_1=observation_data['coords']
        )
        mask_query = self.mask_generator(
            coords_1=query_data['coords'],
        )
        mask_obs_query = self.mask_generator(
            coords_1=query_data['coords'],
            coords_2=observation_data['coords']
        )
        
        # OAT block
        updated_query_token = self.oat_block(
            src=obs_token,
            tgt=query_token,
            src_mask=mask_obs,
            tgt_mask=mask_query,
            memory_mask=mask_obs_query,
            dist_src=...,
            dist_tgt=...,
            dist_memory=...
        )
        
        return self.output_head(
            updated_query_token
        )
        
    def _create_uninformed_tokens(self,
                                 global_data: Dict[str, torch.Tensor],
                                 local_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        global_obs_token = self.global_encoder(
            global_data,
            local_data
        )
        topo_obs_token = self.topographic_encoder(
            local_data['topo']
        )
        uninformed_obs_token = self.fusion_layer1(
            global_obs_token,
            topo_obs_token,
            cond=global_data['time_emb']
        )
        return uninformed_obs_token
        
        
        
                
            
            