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
                        output_var_names=list(model_cfg_fixture.output_head.variable_to_distribution.keys()),
                        lr=1e-3,
                        optimizer_class=torch.optim.AdamW)

    if not compute_loss:
        out = pred(batch_fixture,return_loss_and_targets=False)
    else:
        out, loss, target, target_mask = pred.forward(batch_fixture, return_loss_and_targets=True)


@pytest.mark.parametrize("compile", [True, False])
def test_simple_predictor_train_step(batch_fixture, transform_fixture, model_cfg_fixture, compile):

    output_var_names=list(model_cfg_fixture.output_head.variable_to_distribution.keys())
    input_transform, output_transform = transform_fixture
    
    pred = OATPredictor(oat_model_cfg=model_cfg_fixture,
                        input_transform=input_transform,
                        output_transform=output_transform,
                        output_var_names=output_var_names,
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
                        output_var_names=list(uninformed_model_cfg_fixture.output_head.variable_to_distribution.keys()),
                        lr=1e-3,
                        optimizer_class=torch.optim.AdamW)
    

    pred.training_step(batch=batch_fixture, batch_idx=None)


def test_checkpoint(batch_fixture, transform_fixture, model_cfg_fixture):

    output_var_names=list(model_cfg_fixture.output_head.variable_to_distribution.keys())
    input_transform, output_transform = transform_fixture
    
    pred = OATPredictor(oat_model_cfg=model_cfg_fixture,
                        input_transform=input_transform,
                        output_transform=output_transform,
                        output_var_names=output_var_names,
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
    
    # Verify they are different instances
    assert pred is not reloaded_pred
    
    pl.seed_everything(42)
    prediction = pred(batch_fixture)
    
    pl.seed_everything(42)
    prediction_reloaded = reloaded_pred(batch_fixture)
    # Check that all outputs apart from the samples are equivalent before and 
    # after checkpointing

    def same_distribution(d1, d2):
        if type(d1) != type(d2):
            return False
        
        params1 = d1.__dict__
        params2 = d2.__dict__
        
        if params1.keys() != params2.keys():
            return False
        
        for k in params1:
            v1, v2 = params1[k], params2[k]
            if torch.is_tensor(v1):
                if not torch.allclose(v1, v2):
                    return False
            else:
                if v1 != v2:
                    return False
        return True

    # assert predicted distributions are the same (same parametrization)
    for var_name in output_var_names:
        assert same_distribution(prediction[var_name], prediction_reloaded[var_name])

        assert same_distribution(prediction['pre_oat_output'][var_name], prediction_reloaded['pre_oat_output'][var_name])
    # assert that the samples, given that we forced a seed, are the same before and after checkpointing
    assert torch.allclose(prediction_reloaded['pre_oat_output']['samples'], prediction['pre_oat_output']['samples'])
    assert torch.allclose(prediction_reloaded['pre_oat_output']['samples'], prediction['pre_oat_output']['samples'])
    
    