from oat.predictor_module import OATPredictor
from oat.model_module import OAT
import torch
import os
import pytest

@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipped on CI (requires local datasets)"
)
def test_simple_predictor(batch_fixture):

    model = OAT()
    pred = OATPredictor(oat_model=model,
                        lr=1e-3,
                        optimizer_class=torch.optim.AdamW,
                        training_stations_split=0.3)

    out = pred(batch_fixture,return_loss=False)

    out, loss = pred.forward(batch_fixture, return_loss=True)
