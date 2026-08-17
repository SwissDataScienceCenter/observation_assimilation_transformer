from typing import Any

import pytest
import torch

from oat.data_module.field import Field
from oat.predictor_module.base_predictor import BasePredictor


class _ConcretePredictor(BasePredictor):
    def _normalize_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return batch


class _DummyPredictor(BasePredictor):
    def _normalize_batch(self, batch):
        return batch


def test_base_predictor_var_mappings():
    pred = _ConcretePredictor(output_var_names=["temp", "wind"])
    assert pred.var_to_indx == {"temp": 0, "wind": 1}
    assert pred.indx_to_var == {0: "temp", 1: "wind"}
    assert pred.output_var_names == ["temp", "wind"]


def test_base_predictor_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BasePredictor(output_var_names=["temp"])  # type: ignore[abstract]


def test_transfer_batch_to_device_passthrough_without_field_roles():
    p = _DummyPredictor(output_var_names=["temperature"])
    batch = {"x": torch.zeros(2, 3)}
    out = p.transfer_batch_to_device(batch, torch.device("cpu"), 0)
    assert out["x"].device.type == "cpu"


def test_transfer_batch_to_device_moves_fields_when_field_roles_set():
    p = _DummyPredictor(
        output_var_names=["temperature"],
        field_roles={
            "target": "obs",
            "global": "nwp",
            "geo_bias": "geo",
            "topo": "topo",
        },
    )
    coords = torch.zeros(2, 3, 3)
    field = Field(coords, torch.randn(2, 3, 1), ["obs"], torch.ones(2, 3, 1))
    batch = {
        "observation_data": {"fields": {"obs": field}},
        "index": torch.tensor([0, 1]),
    }
    out = p.transfer_batch_to_device(batch, torch.device("cpu"), 0)
    assert out["observation_data"]["fields"]["obs"].coords.device.type == "cpu"
    assert out["index"].device.type == "cpu"
