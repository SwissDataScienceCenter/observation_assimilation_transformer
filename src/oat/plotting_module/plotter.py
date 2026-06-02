from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from oat.predictor_module.oat_predictor import OATPredictor


class OATPlotter:
    """Wraps an OATPredictor and a dataset to produce diagnostic plots.

    Args:
        predictor: A loaded OATPredictor (or compatible duck-typed object).
        dataset: Dataset instance with `station_coords`, `obs_tensor`,
            `obs_mask_tensor`, `times`, `__getitem__`, and `get_query()`.
        obs_station_indices: Station indices (into dataset.stations) used as
            observation context for every forward pass. If None, all stations
            in the dataset are used.
        n_samples: Number of samples drawn from the predictive distribution
            for every plot. Default 100.
        device: Torch device for inference (e.g. "cpu", "cuda").
    """

    def __init__(
        self,
        predictor: OATPredictor,
        dataset,
        obs_station_indices: Sequence[int] | None = None,
        n_samples: int = 100,
        device: str = "cpu",
    ) -> None:
        self.predictor = predictor
        self.dataset = dataset
        if obs_station_indices is None:
            self.obs_indices: list[int] = list(range(dataset.station_coords.shape[0]))
        else:
            self.obs_indices = list(obs_station_indices)
        self.n_samples = n_samples
        self.device = torch.device(device)

    def _run_inference(self, time_idx: int, query_coords: torch.Tensor) -> torch.Tensor:
        """Run one forward pass and return unnormalized samples.

        Args:
            time_idx: Index into `dataset.times`.
            query_coords: Shape [N_q, 2] in the dataset's coordinate frame.

        Returns:
            Samples tensor of shape [N_q, n_samples, n_vars] (unnormalized).
        """
        raw = self.dataset[time_idx]

        obs_raw = raw["observation_data"]
        obs_data = {k: v[self.obs_indices] for k, v in obs_raw.items()}

        query_data = self.dataset.get_query(query_coords)

        def _add_batch(d: dict) -> dict:
            return {k: v.unsqueeze(0).to(self.device) for k, v in d.items()}

        batch = {
            "global_data": _add_batch(raw["global_data"]),
            "observation_data": _add_batch(obs_data),
            "query_data": _add_batch(query_data),
        }

        batch = self.predictor._normalize_batch(batch)

        original_n_samples = self.predictor.model.num_samples
        self.predictor.model.num_samples = self.n_samples
        try:
            with torch.no_grad():
                y_hat = self.predictor(batch, return_loss_and_targets=False)
        finally:
            self.predictor.model.num_samples = original_n_samples

        # y_hat["samples"]: [1, N_q, n_samples, n_vars]
        samples = self.predictor.output_transform.inverse(y_hat["samples"])
        return samples[0]  # [N_q, n_samples, n_vars]
