from oat.model_module import output_head
import torch
import pytest
    

@pytest.mark.parametrize("observation_assimilation_as", ["innovation", "target"], ids=["innovation", "target"])
def test_output_head(observation_assimilation_as):
    
    batch_size = 32
    n_stations = 12
    in_features = 10
    
    x = torch.randn(batch_size, n_stations, in_features)

    variable_to_distribution = {
        "var1": "normal",
        "var2": "lognormal",
        "var3": "zero_inflated_lognormal",
    }
    
    loss_scaling = torch.tensor([1.0, 0.5, 2.0])
    
    head = output_head.OutputHead(
        in_features=in_features,
        variable_to_distribution=variable_to_distribution,
        loss_scaling=loss_scaling,
        observation_assimilation_as=observation_assimilation_as
    )
    
    pred_distributions = head.forward(x, as_dict=True)
    assert set(pred_distributions.keys()) == set(variable_to_distribution.keys())
    for var_name, dist in pred_distributions.items():
        assert isinstance(dist, output_head.OATDistribution)
        assert dist.batch_shape == (batch_size, n_stations)
        assert dist.event_shape == () 
        
    num_samples = 5
    samples_dict = head.sample(x, num_samples=num_samples, differentiable=False, as_dict=True)
    
    for var_name, samples in samples_dict.items():
        assert samples.shape == (batch_size, n_stations, num_samples, 1)
    
    samples_tensor = head.sample(x, num_samples=num_samples, differentiable=False, as_dict=False)
    
    assert samples_tensor.shape == (batch_size, n_stations, num_samples, len(variable_to_distribution))
    
    
    target = samples_tensor[...,0,:] # get a target that fits the support 
    out, loss, per_var_losses = head.loss(x=x, target=target, mask=torch.ones_like(target))
    
    assert isinstance(loss, torch.Tensor)
    assert isinstance(out, dict)
    assert loss.shape == ()
    
    
    sum = 0.0
    for per_var_loss in per_var_losses.values():
        assert isinstance(per_var_loss, torch.Tensor)
        assert per_var_loss.shape == ()
        
        sum += per_var_loss
        
    assert torch.isclose(sum, loss)