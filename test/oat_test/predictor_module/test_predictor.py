from oat.predictor_module import OATPredictor
from oat.model_module import OAT
import torch
import os
import pytest
from pytorch_lightning.loggers.logger import DummyLogger

@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipped on CI (requires local datasets)"
)
@pytest.mark.parametrize("compute_loss", [True, False])
def test_simple_predictor_forward(batch_fixture, transform_fixture, compute_loss):

    model = OAT()
    
    input_transform, output_transform = transform_fixture
    
    pred = OATPredictor(oat_model=model,
                        input_transform=input_transform,
                        output_transform=output_transform,
                        lr=1e-3,
                        optimizer_class=torch.optim.AdamW,
                        training_stations_split=0.3)

    if not compute_loss:
        out = pred(batch_fixture,return_loss=False)
    else:
        out, loss = pred.forward(batch_fixture, return_loss_and_targets=True)

@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipped on CI (requires local datasets)"
)    
def test_simple_predictor_train_step(batch_fixture, transform_fixture):
    
    model = OAT()
    input_transform, output_transform = transform_fixture
    
    pred = OATPredictor(oat_model=model,
                        input_transform=input_transform,
                        output_transform=output_transform,
                        lr=1e-3,
                        optimizer_class=torch.optim.AdamW,
                        training_stations_split=0.3)
    
    pred.training_step(batch=batch_fixture, batch_idx=None)
