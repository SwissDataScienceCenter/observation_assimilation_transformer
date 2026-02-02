import numpy as np
import torch

from oat.data_module.statistics import RunningFeatureStats


def test_statistics():

    stats = RunningFeatureStats(num_features=4, feature_dim=1)

    data = torch.randn(128, 4, 10,12)

    data[:,0] += 10
    data[:,0] *= 5

    data[:,1] += -13
    data[:,1] *= 3

    data[:,2] += 1000
    data[:,2] *= 150

    data[:,3] += -200
    data[:,3] *= 42

    stride = 8
    for i in range(0,128, stride):

        x = data[i:i+stride]

        stats.update(x=x)

    assert torch.allclose(stats.std, data.std(dim=[0,2,3], unbiased=False), atol=1e-5)
    assert torch.allclose(stats.mean, data.mean(dim=[0,2,3]), atol=1e-5)

    assert (data.max(dim=0).values.max(dim=1).values.max(dim=1).values == stats.max).all()
    assert (data.min(dim=0).values.min(dim=1).values.min(dim=1).values == stats.min).all()

    # Reset
    stats.reset()
    assert (stats.mean == 0.0).all()
    assert (stats.M2 == 0.0).all()
    assert (stats.count == 0.0).all()
    assert (stats.max.isneginf()).all()
    assert (stats.min.isinf()).all()
    assert (stats.std == 0.0).all()


    # Introduce NaNs
    nan_mask = torch.rand_like(data) < 0.05
    data[nan_mask] = float('nan')

    for i in range(0,128, stride):

        x = data[i:i+stride]

        stats.update(x=x)

    data_np = data.cpu().numpy()

    mean_ref = np.nanmean(data_np, axis=(0,2,3))
    std_ref = np.nanstd(data_np, axis=(0,2,3))
    min_ref = np.nanmin(data_np, axis=(0,2,3))
    max_ref = np.nanmax(data_np, axis=(0,2,3))

    assert torch.allclose(stats.mean, torch.Tensor(mean_ref), atol=1e-5)
    assert torch.allclose(stats.std, torch.Tensor(std_ref), atol=1e-5)
    assert (stats.min == min_ref).all()
    assert (stats.max == max_ref).all()
    

def test_get_transforms():

    stats = RunningFeatureStats(num_features=4, feature_dim=1)

    data = torch.randn(128, 4, 10,12)

    data[:,0] += 10
    data[:,0] *= 5

    data[:,1] += -13
    data[:,1] *= 3

    data[:,2] += 1000
    data[:,2] *= 150

    data[:,3] += -200
    data[:,3] *= 42

    stride = 8
    for i in range(0,128, stride):

        x = data[i:i+stride]

        stats.update(x=x)

    standard_t = stats.get_standard_transform()

    # assert normalization works
    assert torch.isclose(standard_t(data).mean(), torch.tensor(0.0), atol=1e-2)
    # assert inverse normalization works
    assert torch.allclose(standard_t.inverse(standard_t(data)), data)
    
    
    minmax_t = stats.get_minmax_transform()
    
    minmaxed_data = minmax_t(data)
    # assert each feature is in [0.0,1.0]
    for f_i in range(data.shape[1]):
        assert minmaxed_data[:,f_i].min() == 0.0
        assert minmaxed_data[:,f_i].max() == 1.0



