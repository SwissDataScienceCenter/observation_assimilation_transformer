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


def test_closures():

    stats = RunningFeatureStats(num_features=4, feature_dim=1)

    data = torch.randn(128, 4, 10,12)
    data += 100
    data *= 5

    stride = 8
    for i in range(0,128, stride):

        x = data[i:i+stride]

        stats.update(x=x)

    normalizers = stats.get_standard_normalizer_closures()
    normalizer = normalizers['normalizer']
    denormalizer = normalizers['denormalizer']


    sample_data = torch.randn(32, 4, 10,12)
    sample_data += 100
    sample_data *= 5

    normalized_sample = normalizer(sample_data)

    # Assert normalization works
    torch.isclose(normalized_sample.mean(), torch.tensor(0.0), atol=1e-2)

    # Assert denormalization works
    assert torch.allclose(denormalizer(normalized_sample), sample_data, atol=1e-6)

    # Assert that changing the statistics, does not affect the closure's memory
    # ie that there are no side-effects

    # new data with new stats
    new_data = torch.randn(128, 4, 10,12)
    new_data += 150
    new_data *= 7

    # update running stats
    for i in range(0,128, stride):

        x = new_data[i:i+stride]
        stats.update(x=x)

    # normalizer comes from stats,
    # but stats changed. This makes sure that there are
    # no side effects due to values saved by reference
    new_normalized_sample = normalizer(sample_data)

    assert (new_normalized_sample == normalized_sample).all()


