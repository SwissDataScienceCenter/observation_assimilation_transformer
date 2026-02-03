from typing import Callable

import torch

from oat.data_module.transform import (
    MinMaxTransform,
    StandardNormalizeTransform,
    Transform,
)


class RunningFeatureStats:

    def __init__(self, num_features: int, feature_dim: int, device: str = 'cpu'):
        """
        This class computes running mean, std, min and max for large datasets
        without the need to load the whole dataset.
        Make sure that the arguments num_features and feature_dim include a batch dimension.

        Args:
            num_features: the number of features in the data
            feature_dim: the dimension where the features are stored
            device: the device on which running statistics and input data are stored
        """
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.device = device

        # Running statistics
        self.count = torch.zeros(num_features, device=device)
        self.mean = torch.zeros(num_features, device=device)
        self.M2 = torch.zeros(num_features, device=device)
        self.min = torch.full((num_features,), float('inf'), device=device)
        self.max = torch.full((num_features,), float('-inf'), device=device)

    def update(self, x: torch.Tensor) -> None:

        # Ensure input is on the same device as the running statistics
        if x.device != self.count.device:
            x = x.to(self.count.device)
        # Move feature dimension to first axis and flatten remaining dims
        dims = list(range(x.dim()))
        if self.feature_dim < 0:
            feature_dim = x.dim() + self.feature_dim
        else:
            feature_dim = self.feature_dim
        permute_order = [feature_dim] + [i for i in dims if i != feature_dim]
        x_flat = x.permute(*permute_order).contiguous().view(self.num_features, -1)

        # Mask NaNs
        mask = ~torch.isnan(x_flat)
        valid_counts = mask.sum(dim=1)  # number of non-NaN elements per feature
        valid_mask = valid_counts > 0
        if valid_mask.sum() == 0:
            return  # nothing to update

        # Min/max update
        masked_min = torch.where(mask, x_flat, torch.tensor(float('inf'), device=x.device))
        self.min[valid_mask] = torch.min(self.min[valid_mask], masked_min[valid_mask].min(dim=1).values)

        masked_max = torch.where(mask, x_flat, torch.tensor(float('-inf'), device=x.device))
        self.max[valid_mask] = torch.max(self.max[valid_mask], masked_max[valid_mask].max(dim=1).values)

        # Mean/variance Welford algorithm per feature
        for f in range(self.num_features):
            if valid_counts[f] == 0:
                continue
            vals = x_flat[f, mask[f]]
            bm = vals.mean()
            bv = vals.var(unbiased=False)
            bc = vals.numel()

            delta = bm - self.mean[f]
            total_count = self.count[f] + bc

            self.mean[f] += delta * bc / total_count
            self.M2[f] += bv * bc + delta**2 * self.count[f] * bc / total_count
            self.count[f] = total_count

    @property
    def variance(self) -> torch.Tensor:
        return self.M2 / torch.clamp(self.count, min=1)

    @property
    def std(self) -> torch.Tensor:
        return self.variance.sqrt()

    def get_standard_transform(self) -> Transform:
        return StandardNormalizeTransform(mean=self.mean.clone(),
                                          std=self.std.clone(),
                                          channel_dim=self.feature_dim)

    def get_minmax_transform(self) -> Transform:
        return MinMaxTransform(data_min=self.min.clone(),
                               data_max=self.max.clone(),
                               channel_dim=self.feature_dim)


    def reset(self) -> None:
        """
        Reset all running statistics
        """
        self.count.zero_()
        self.mean.zero_()
        self.M2.zero_()
        self.min.fill_(float('inf'))
        self.max.fill_(float('-inf'))
