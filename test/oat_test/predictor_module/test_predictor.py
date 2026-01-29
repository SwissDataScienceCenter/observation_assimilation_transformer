from oat.predictor_module import OATPredictor
import torch

def test_simple_predictor(batch_fixture):

    pred = OATPredictor(oat_model=None,
                        lr=1e-3,
                        optimizer_class=torch.optim.AdamW)
    
    
    pred(batch_fixture)