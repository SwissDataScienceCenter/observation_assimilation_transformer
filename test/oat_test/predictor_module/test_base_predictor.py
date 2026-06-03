from typing import Any

import pytest
import torch

from oat.predictor_module.base_predictor import BasePredictor


class _ConcretePredictor(BasePredictor):
    def _normalize_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return batch

    def predict_samples(self, batch: dict[str, Any], n_samples: int) -> torch.Tensor:
        return torch.zeros(1, 1, n_samples, len(self.output_var_names))


def test_base_predictor_var_mappings():
    pred = _ConcretePredictor(output_var_names=["temp", "wind"])
    assert pred.var_to_indx == {"temp": 0, "wind": 1}
    assert pred.indx_to_var == {0: "temp", 1: "wind"}
    assert pred.output_var_names == ["temp", "wind"]


def test_base_predictor_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BasePredictor(output_var_names=["temp"])  # type: ignore[abstract]
