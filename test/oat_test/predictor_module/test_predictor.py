import tempfile
import pytorch_lightning as pl

from oat.predictor_module import OATPredictor
from oat.model_module import OAT
import torch
import os
import pytest
from pytorch_lightning.loggers.logger import DummyLogger


@pytest.mark.parametrize("compute_loss", [True, False])
def test_simple_predictor_forward(batch_fixture, transform_fixture, compute_loss, model_cfg_fixture):
    
    input_transform, output_transform = transform_fixture
    
    pred = OATPredictor(oat_model_cfg=model_cfg_fixture,
                        input_transform=input_transform,
                        output_transform=output_transform,
                        output_var_names=[str(i) for i in range(model_cfg_fixture.output_head.target_features)],
                        lr=1e-3,
                        optimizer_class=torch.optim.AdamW)

    if not compute_loss:
        out = pred(batch_fixture,return_loss_and_targets=False)
    else:
        out, loss, target, target_mask = pred.forward(batch_fixture, return_loss_and_targets=True)


@pytest.mark.parametrize("compile", [True, False])
def test_simple_predictor_train_step(batch_fixture, transform_fixture, model_cfg_fixture, compile):


    input_transform, output_transform = transform_fixture
    
    pred = OATPredictor(oat_model_cfg=model_cfg_fixture,
                        input_transform=input_transform,
                        output_transform=output_transform,
                        output_var_names=[str(i) for i in range(model_cfg_fixture.output_head.target_features)],
                        lr=1e-3,
                        optimizer_class=torch.optim.AdamW)
    
    if compile:
        pred = torch.compile(pred)
    
    pred.training_step(batch=batch_fixture, batch_idx=None)



def test_uninformed_predictor_train_step(batch_fixture, transform_fixture, uninformed_model_cfg_fixture):


    input_transform, output_transform = transform_fixture
    
    pred = OATPredictor(oat_model_cfg=uninformed_model_cfg_fixture,
                        input_transform=input_transform,
                        output_transform=output_transform,
                        output_var_names=[str(i) for i in range(uninformed_model_cfg_fixture.output_head.target_features)],
                        lr=1e-3,
                        optimizer_class=torch.optim.AdamW)
    

    pred.training_step(batch=batch_fixture, batch_idx=None)


def test_checkpoint(batch_fixture, transform_fixture, model_cfg_fixture):


    input_transform, output_transform = transform_fixture
    
    pred = OATPredictor(oat_model_cfg=model_cfg_fixture,
                        input_transform=input_transform,
                        output_transform=output_transform,
                        output_var_names=[str(i) for i in range(model_cfg_fixture.output_head.target_features)],
                        lr=1e-3,
                        optimizer_class=torch.optim.AdamW)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = f"{tmpdir}/model.ckpt"

        # Manually create a Lightning checkpoint
        torch.save(
            {
                'state_dict': pred.state_dict(),
                'hyper_parameters': pred.hparams,
                'pytorch-lightning_version': pl.__version__
            },
            ckpt_path
        )

        # Reload the module
        reloaded_pred = type(pred).load_from_checkpoint(ckpt_path)
       
    # Remove randomness like dropouts 
    pred = pred.eval()
    reloaded_pred = reloaded_pred.eval()
    
    prediction = pred(batch_fixture)
    prediction_reloaded = reloaded_pred(batch_fixture)
    # Check that all outputs apart from the samples are equivalent before and 
    # after checkpointing
    for k in [key for key in prediction.keys() if key != "samples"]:
        if k == "pre_oat_output":
            for kk in [key for key in prediction[k].keys() if key != "samples"]:
                assert (prediction[k][kk] == prediction_reloaded[k][kk]).all()
        else:
            assert (prediction[k] == prediction_reloaded[k]).all()
    
    