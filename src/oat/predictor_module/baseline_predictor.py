from oat.model_module import baseline
import pytorch_lightning as pl
from typing import Dict, Any
import torch


class BaselinePredictor(pl.LightningModule):
    
    def __init__(self, config: baseline.InterpolaterBaselineConfig) -> None:
        super().__init__()
        self.model = baseline.InterpolaterBaseline(config)
        self.hparams.update(self.model.interpolater._interpolation_params)
        
    def forward(self, global_data: Dict[str, Any], 
        observation_data: Dict[str, Any],
        query_data: Dict[str, Any]) -> torch.Tensor:
        return self.model(global_data, observation_data, query_data)
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        y_pred = self(batch['global_data'], batch['observation_data'], batch['query_data'])
        y_pred = y_pred.unsqueeze(-2)  # [B, S, D] -> [B, S, 1, D]
        
        self._log_instant_metrics(y_hat_samples=y_pred, 
                                  y_true=batch['query_data']['target'],
                                  mask=batch['query_data']['mask'],
                                  prefix="val")
        return y_pred
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        y_pred = self(batch['global_data'], batch['observation_data'], batch['query_data'])
        y_pred = y_pred.unsqueeze(-2)  # [B, S, D] -> [B, S, 1, D]
        
        self._log_instant_metrics(y_hat_samples=y_pred, 
                                  y_true=batch['query_data']['target'],
                                  mask=batch['query_data']['mask'],
                                  prefix="test")
        return y_pred
    
    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        y_pred = self(batch['global_data'], batch['observation_data'], batch['query_data'])
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
    
        