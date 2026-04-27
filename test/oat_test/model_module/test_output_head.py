from oat.model_module import output_head
import torch
import pytest


def test_linear_output_head() -> None:
    batch_dim = 64
    target_dim = 2
    input_dim = 10
    n_stations = 23
    num_samples = 10

    target = torch.randn(batch_dim, n_stations, target_dim)
    input_tensor = torch.randn(batch_dim, n_stations, input_dim)
    mask = torch.randint(0, 2, (batch_dim, n_stations, target_dim)).bool()
    for loss_fn in ['mse', 'mae', 'huber']:
        linear_output_head = output_head.LinearOutputHead(
            in_features=input_dim, target_features=target_dim, loss_fn=loss_fn
        )
        out = linear_output_head(input_tensor)  # Forward pass
        preds = out['val_pred']
        assert preds.shape == (batch_dim, n_stations, target_dim)  # Check output
        inno = linear_output_head.compute_innovation(input_tensor, target, mask)  # Innovation computation
        assert inno.shape == (batch_dim, n_stations, target_dim)  # Check
        _, loss = linear_output_head.loss(input_tensor, target, mask)  # Loss computation
        assert loss.dim() == 0  # Check that loss is a scalar
        assert loss >= 0.0  # Loss should be non-negative
        samples = linear_output_head.sample(out, num_samples)  # Sampling from output distribution
        assert samples.shape == (batch_dim, n_stations, num_samples, target_dim)  # Check sample shape
        assert torch.allclose(samples, preds.unsqueeze(-2).expand(-1, -1, num_samples, -1))  # Check that all samples match predictions

@pytest.mark.parametrize("loss_fn", ['nll', 'crps'])
def test_gaussian_output_head(loss_fn) -> None:
    batch_dim = 64
    target_dim = 2
    input_dim = 10
    n_stations = 23
    num_samples = 10

    target = torch.randn(batch_dim, n_stations, target_dim)
    input_tensor = torch.randn(batch_dim, n_stations, input_dim)
    mask = torch.randint(0, 2, (batch_dim, n_stations, target_dim)).bool()
    for std_fn in ['softplus', 'exp']:
        gaussian_output_head = output_head.GaussianOutputHead(
            in_features=input_dim, target_features=target_dim, std_fn=std_fn, loss_fn=loss_fn
        )
        out = gaussian_output_head(input_tensor)  # Forward pass
        assert out["mu_pred"].shape == (batch_dim, n_stations, target_dim)  # Check mean output shape
        assert out["std_pred"].shape == (batch_dim, n_stations, target_dim)  # Check std output shape
        inno = gaussian_output_head.compute_innovation(input_tensor, target, mask)  # Innovation computation
        assert inno.shape == (batch_dim, n_stations, target_dim)  # Check
        _, loss = gaussian_output_head.loss(input_tensor, target, mask)  # Loss computation
        assert loss.dim() == 0  # Check that loss is a scalar
        samples = gaussian_output_head.sample(out, num_samples)  # Sampling from output distribution
        assert samples.shape == (batch_dim, n_stations, num_samples, target_dim)
    
def test_quantile_output_head() -> None:
    batch_dim = 64
    target_dim = 2
    input_dim = 10
    n_stations = 23
    num_samples = 10
    
    target = torch.randn(batch_dim, n_stations, target_dim)
    input_tensor = torch.randn(batch_dim, n_stations, input_dim)
    mask = torch.randint(0, 2, (batch_dim, n_stations, target_dim)).bool()
    quantile_output_head = output_head.QuantileOutputHead(
        in_features=input_dim, target_features=target_dim, degree=50
    )
    out = quantile_output_head(input_tensor)  # Forward pass
    qvals = out["qval_pred"]
    assert qvals.shape == (batch_dim, n_stations, len(quantile_output_head.taus), target_dim)  # Check output
    assert torch.diff(quantile_output_head.taus).min() > 0  # Check taus are increasing
    inno = quantile_output_head.compute_innovation(input_tensor, target, mask)  # Innovation computation
    assert inno.shape == (batch_dim, n_stations, target_dim)  # Check
    _, loss = quantile_output_head.loss(input_tensor, target, mask)  # Loss computation
    assert loss.dim() == 0  # Check that loss is a scalar
    assert loss >= 0.0  # Loss should be non-negative
    mask0 = torch.zeros(batch_dim, n_stations, target_dim).bool()  # All masked
    _, loss_all_masked = quantile_output_head.loss(input_tensor, target, mask0)  # Loss computation
    assert loss_all_masked.dim() == 0  # Check that loss is a scalar
    assert loss_all_masked == 0.0  # Loss should be zero when all values are masked
    mask1 = torch.ones(batch_dim, n_stations, target_dim).bool()  # None masked
    _, loss_none_masked = quantile_output_head.loss(input_tensor, target, mask1)  # Loss computation
    assert loss_none_masked.dim() == 0  # Check that loss is a scalar
    assert loss_none_masked >= 0.0  # Loss should be non-negative
    samples = quantile_output_head.sample(out, num_samples)  # Sampling from output distribution
    assert samples.shape == (batch_dim, n_stations, num_samples, target_dim)
    
    
def test_output_head():
    
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
        loss_scaling=loss_scaling
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
    loss = head.loss(x=x, target=target, mask=torch.ones_like(target))
        

    
# def test_multiple_univariate_heads() -> None:
    # 
    # batch_dim = 64

    # input_dim = 10
    # n_stations = 23
    
    # input_tensor = torch.randn(batch_dim, n_stations, input_dim)
    
    # heads_config = {
    #     "T_2M": (output_head.LinearOutputHead, {"loss_fn": "mse"}),
    #     "V_10": (output_head.GaussianOutputHead, {"std_fn": "softplus", "loss_fn": "crps"}),
    #     "U_10": (output_head.GaussianOutputHead, {"std_fn": "softplus", "loss_fn": "crps"}),
    # }
    
    # multi_head = output_head.MultipleUnivariateHeads(in_features=input_dim, heads=heads_config)
    # out = multi_head(input_tensor)  # Forward pass
    # assert set(out.keys()) == set(heads_config.keys())  # Check all heads are present
    # assert out["T_2M"]["val_pred"].shape == (batch_dim, n_stations, 1)
    # assert out["V_10"]["mu_pred"].shape == (batch_dim, n_stations, 1)  
    # assert out["V_10"]["std_pred"].shape == (batch_dim, n_stations, 1) 
    # assert out["U_10"]["mu_pred"].shape == (batch_dim, n_stations, 1)  
    # assert out["U_10"]["std_pred"].shape == (batch_dim, n_stations, 1) 
    
    # num_samples = 11
    # dict_samples = multi_head.sample(out, num_samples)
    
    # # assert that for each variable we have 11 univariate samples
    # assert set(dict_samples.keys()) == set(heads_config.keys())
    # assert dict_samples["T_2M"].shape == (batch_dim, n_stations, num_samples, 1)
    # assert dict_samples["V_10"].shape == (batch_dim, n_stations, num_samples, 1)
    # assert dict_samples["U_10"].shape == (batch_dim, n_stations, num_samples, 1) 
    
    # # assert that we can also get concatenated samples across variables
    # concat_samples = multi_head.sample(out, num_samples, as_dict=False)
    # assert concat_samples.shape == (batch_dim, n_stations, num_samples, len(heads_config))
    
    
    # # test loss
    # target = torch.randn(batch_dim, n_stations, len(heads_config))
    # mask = torch.randint(0, 2, (batch_dim, n_stations, len(heads_config))).bool()
    # outs, total_loss = multi_head.loss(input_tensor, target, mask)
    # assert total_loss.dim() == 0  # Check that loss is a scalar
    # assert total_loss >= 0.0  # Loss should be non-negative