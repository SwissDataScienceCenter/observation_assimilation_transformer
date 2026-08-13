# src/oat/predictor_module/baseline_predictor.py
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from oat.model_module import baseline
from oat.predictor_module.base_predictor import BasePredictor
from oat.predictor_module.field_batch_utils import _unwrap_fields_to_model_batch
from oat.predictor_module.functional_metrics import point_mae


class NWPInterpolatorPredictor(BasePredictor):
    """Wraps ``NWPInterpolator`` for the Field data path: no parameters, no training.

    Works in raw data units end to end (no input_transform / output_transform):
    the model already emits obs-unit predictions, compared directly against
    the raw obs pulled off the field batch.
    """

    def __init__(
        self,
        config: baseline.NWPInterpolatorConfig,
        output_var_names: List[str],
        field_roles: Dict[str, str],
    ) -> None:
        super().__init__(output_var_names=output_var_names, field_roles=field_roles)
        self.model = baseline.NWPInterpolator(config)
        self.metrics_fn = {"point_mae": point_mae}

    def _normalize_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return _unwrap_fields_to_model_batch(
            batch, self.field_roles, self.model.requires_observation
        )

    def forward(
        self, batch: Dict[str, Any], query_locations: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Any, Any]:
        if query_locations is not None:
            raise NotImplementedError("Custom query_locations not yet supported")
        q = batch["query_data"]
        return self.model(q["global_tokens"]), q.get("obs"), q.get("mask_obs")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch = self._normalize_batch(batch)
        y_pred, target_obs, target_mask_obs = self(batch)
        y_pred = y_pred.unsqueeze(-2)  # [B, S, D] -> [B, S, 1, D]

        self._log_instant_metrics(
            y_hat_samples=y_pred,
            y_true=target_obs,
            mask=target_mask_obs,
            prefix="val",
        )
        return y_pred

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch = self._normalize_batch(batch)
        y_pred, target_obs, target_mask_obs = self(batch)
        y_pred = y_pred.unsqueeze(-2)  # [B, S, D] -> [B, S, 1, D]

        self._log_instant_metrics(
            y_hat_samples=y_pred,
            y_true=target_obs,
            mask=target_mask_obs,
            prefix="test",
        )
        return y_pred

    def predict_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
        return_as: Literal["samples", "distribution"] = "samples",
        n_samples: Optional[int] = None,
    ) -> torch.Tensor:
        batch = self._normalize_batch(batch)
        if return_as == "samples":
            with torch.no_grad():
                y_pred, _, _ = self(batch)  # [B, N_q, n_vars]
            n = n_samples if n_samples is not None else 1
            return y_pred.unsqueeze(-2).expand(-1, -1, n, -1)  # [B, N_q, n_samples, n_vars]
        elif return_as == "distribution":
            raise ValueError(
                "NWPInterpolatorPredictor is deterministic; "
                "'distribution' return_as is not supported."
            )
        else:
            raise ValueError(
                f"Invalid value for return_as: {return_as!r}. "
                "Expected 'samples' or 'distribution'."
            )

    def _log_instant_metrics(
        self,
        y_hat_samples: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
        prefix: str = "",
    ) -> None:
        for metric_name, metric_fn in self.metrics_fn.items():
            metric_values = metric_fn(
                y_target=y_true, y_hat=y_hat_samples, mask=mask, aggregation="variable"
            )
            for i, metric_val in enumerate(metric_values):
                var_name = self.indx_to_var[i]
                self.log(
                    name=f"{prefix}_{metric_name}_{var_name}",
                    value=metric_val,
                    prog_bar=True,
                    on_step=True,
                    on_epoch=True,
                )


def create_baseline_predictor(dataset: Any) -> NWPInterpolatorPredictor:
    """Build the whole NWP-interpolator predictor from a Field-path dataset."""
    fc = dataset.field_config
    roles = fc.roles
    obs_out = list(fc.fields[roles["target"]].out)
    config = baseline.NWPInterpolatorConfig(
        nwp_names=list(fc.fields[roles["global"]].out),
        obs_names=obs_out,
        obs_to_nwp=dataset.obs_to_nwp,
    )
    return NWPInterpolatorPredictor(
        config=config, output_var_names=obs_out, field_roles=dict(roles)
    )
