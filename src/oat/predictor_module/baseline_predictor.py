from typing import Callable, Dict, Any, List, Optional, Tuple

import torch

from oat.model_module import baseline
from oat.predictor_module.base_predictor import BasePredictor
from oat.predictor_module.functional_metrics import point_mae


class NWPInterpolatorPredictor(BasePredictor):
    
    def __init__(self,
                 config: baseline.NWPInterpolatorConfig,
                 output_var_names: List,
                 input_transform: Callable | None = None) -> None:
        super().__init__(output_var_names=output_var_names)
        self.model = baseline.NWPInterpolator(config)
        self.hparams.update(self.model.interpolator._interpolation_params)
        self.input_transform = input_transform

        self.metrics_fn = {
            'point_mae': point_mae
        }
        
    def forward(self, batch: Dict[str, Any], query_locations: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Any, Any]:
        if query_locations is None:
            target_obs = batch['query_data'].get('obs', None)
            target_mask_obs = batch['query_data'].get('mask_obs', None)
            query_data = {k: v for k, v in batch['query_data'].items() if k not in ('obs', 'mask_obs')}
        else:
            # TODO implement
            raise NotImplementedError("Custom query_locations not yet supported")
        out = self.model(batch['global_data'], query_data)
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

    def _normalize_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return batch

    def predict_samples(self, batch: Dict[str, Any], n_samples: int) -> torch.Tensor:
        """NWP interpolation is deterministic; all n_samples slices are identical."""
        if self.input_transform is not None:
            batch = self.input_transform(batch)
        with torch.no_grad():
            out, _, _ = self(batch)  # [B, N_q, n_vars]
        return out.unsqueeze(2).expand(-1, -1, n_samples, -1)

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


def create_baseline_predictor(
    config: baseline.NWPInterpolatorConfig,
    obs_vars: List[str],
    pre_transform: Callable | None = None,
) -> NWPInterpolatorPredictor:
    """Build a BaselinePredictor from a config and dataset-provided pre-transform."""
    return NWPInterpolatorPredictor(
        config=config,
        output_var_names=obs_vars,
        input_transform=pre_transform,
    )