from oat.model_module import output_head
import torch


def test_linear_output_head() -> None:
    batch_dim = 64
    target_dim = 2
    input_dim = 10
    n_stations = 23

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
    
def test_gaussian_output_head() -> None:
    batch_dim = 64
    target_dim = 2
    input_dim = 10
    n_stations = 23

    target = torch.randn(batch_dim, n_stations, target_dim)
    input_tensor = torch.randn(batch_dim, n_stations, input_dim)
    mask = torch.randint(0, 2, (batch_dim, n_stations, target_dim)).bool()
    for std_fn in ['softplus', 'exp']:
        gaussian_output_head = output_head.GaussianOutputHead(
            in_features=input_dim, target_features=target_dim, std_fn=std_fn
        )
        out = gaussian_output_head(input_tensor)  # Forward pass
        assert out["mu_pred"].shape == (batch_dim, n_stations, target_dim)  # Check mean output shape
        assert out["std_pred"].shape == (batch_dim, n_stations, target_dim)  # Check std output shape
        inno = gaussian_output_head.compute_innovation(input_tensor, target, mask)  # Innovation computation
        assert inno.shape == (batch_dim, n_stations, target_dim)  # Check
        _, loss = gaussian_output_head.loss(input_tensor, target, mask)  # Loss computation
        assert loss.dim() == 0  # Check that loss is a scalar
    
def test_quantile_output_head() -> None:
    batch_dim = 64
    target_dim = 2
    input_dim = 10
    n_stations = 23

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