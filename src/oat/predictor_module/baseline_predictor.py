from oat.model_module import baseline
import pytorch_lightning as pl
from typing import Dict, Any, Callable, Optional, List
import torch
from oat.predictor_module.functional_metrics import sample_mae, point_mae


class BaselinePredictor(pl.LightningModule):
    
    def __init__(self, 
                 config: baseline.InterpolaterBaselineConfig,
                 output_var_names: List,
                 input_transform: Callable | None = None) -> None:
        super().__init__()
        self.model = baseline.InterpolaterBaseline(config)
        self.hparams.update(self.model.interpolater._interpolation_params)
        self.input_transform = input_transform
        
        self.metrics_fn = {
            'point_mae': point_mae
        }
        
        self.output_var_names = output_var_names
        self.var_to_indx = {v: i for i,v in enumerate(self.output_var_names)}
        self.indx_to_var = {i: v for i,v in enumerate(self.output_var_names)}
        
    def forward(self, batch: Dict[str, Any], query_locations: Optional[torch.Tensor] = None) -> torch.Tensor:
        if query_locations is None:
            target_obs = batch['query_data'].get('obs', None)
            target_mask_obs = batch['query_data'].get('mask_obs', None)
            query_data = {k: v for k, v in batch['query_data'].items() if k not in ('obs', 'mask_obs')}
        else:
            # TODO implement
            raise NotImplementedError("Custom query_locations not yet supported")
        out = self.model(batch['global_data'], batch['observation_data'], query_data)
        return out, target_obs, target_mask_obs
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch = self.input_transform(batch) if self.input_transform is not None else batch
        y_pred, target_obs, target_mask_obs = self(batch)
        y_pred = y_pred.unsqueeze(-2)  # [B, S, D] -> [B, S, 1, D]
        
        self._log_instant_metrics(y_hat_samples=y_pred, 
                                  y_true=target_obs,
                                  mask=target_mask_obs,
                                  prefix="val")
        return y_pred
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch = self.input_transform(batch) if self.input_transform is not None else batch
        y_pred, target_obs, target_mask_obs = self(batch)
        y_pred = y_pred.unsqueeze(-2)  # [B, S, D] -> [B, S, 1, D]
        
        self._log_instant_metrics(y_hat_samples=y_pred, 
                                  y_true=target_obs,
                                  mask=target_mask_obs,
                                  prefix="test")
        return y_pred
    
    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch = self.input_transform(batch) if self.input_transform is not None else batch
        y_pred, _, _ = self(batch)
        return y_pred
        
    def _log_instant_metrics(self, y_hat_samples: torch.Tensor, 
                                        y_true: torch.Tensor, 
                                        mask: torch.Tensor,
                                        prefix: str = "") -> None:
        for metric_name, metric_fn in self.metrics_fn.items():
            
            metric_values = metric_fn(y_target=y_true,
                                      y_hat=y_hat_samples,
                                      mask=mask,
                                      aggregation='variable')
            
            for i, metric_val in enumerate(metric_values):
                var_name = self.indx_to_var[i]
                
                self.log(name=f'{prefix}_{metric_name}_{var_name}', 
                         value=metric_val, 
                         prog_bar=True, 
                         on_step=True, 
                         on_epoch=True)
    
        