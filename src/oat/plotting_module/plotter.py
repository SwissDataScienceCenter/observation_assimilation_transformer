from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
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
        eval_indices: Station indices shown as evaluation targets (squares) in
            the spatial plot. If None, falls back to current behaviour (all
            stations not in obs_station_indices are shown as squares).
        n_samples: Number of samples drawn from the predictive distribution
            for every plot. Default 100.
        device: Torch device for inference (e.g. "cpu", "cuda").
    """

    def __init__(
        self,
        predictor: OATPredictor,
        dataset: Any,
        obs_station_indices: Sequence[int] | None = None,
        eval_indices: Sequence[int] | None = None,
        n_samples: int = 100,
        device: str = "cpu",
    ) -> None:
        self.predictor = predictor
        self.dataset = dataset
        if obs_station_indices is None:
            self.obs_indices: list[int] = list(range(dataset.station_coords.shape[0]))
        else:
            self.obs_indices = list(obs_station_indices)
        if eval_indices is None:
            self.eval_indices: list[int] = []
        else:
            self.eval_indices = list(eval_indices)
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

        def _add_batch(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {k: v.unsqueeze(0).to(self.device) for k, v in d.items()}

        batch = {
            "global_data": _add_batch(raw["global_data"]),
            "observation_data": _add_batch(obs_data),
            "query_data": _add_batch(query_data),
        }

        batch = self.predictor._normalize_batch(batch)  # type: ignore[arg-type]

        # OATPredictor.forward() doesn't expose num_samples; mutate and restore.
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

    def plot_timeseries(
        self,
        ax: Any,
        center: tuple[float, float],
        var: str,
        time_indices: list[int],
        quantile_lo: float = 0.05,
        quantile_hi: float = 0.95,
    ) -> None:
        """Plot predicted distribution over time at an arbitrary location.

        Draws a mean line and a shaded uncertainty band onto *ax*.

        Args:
            ax: Matplotlib Axes to draw onto.
            center: Query location as (x, y) in the dataset's coordinate frame.
            var: Name of the variable to plot (must be in predictor.var_to_indx).
            time_indices: Dataset time indices to predict at.
            quantile_lo: Lower quantile for the uncertainty band.
            quantile_hi: Upper quantile for the uncertainty band.
        """
        var_idx = self.predictor.var_to_indx[var]
        query_coords = torch.tensor([center], dtype=torch.float32)  # [1, 2]

        medians, lo_vals, hi_vals = [], [], []

        for t_idx in time_indices:
            samples = self._run_inference(t_idx, query_coords)  # [1, n_samples, n_vars]
            var_samples = samples[0, :, var_idx]  # [n_samples]
            medians.append(var_samples.median().item())
            std = var_samples.std().item()
            lo_vals.append(var_samples.quantile(quantile_lo).item())
            hi_vals.append(var_samples.quantile(quantile_hi).item())

        times = [self.dataset.times[i] for i in time_indices]

        ax.plot(times, medians, label="median")
        ax.fill_between(
            times,
            lo_vals,
            hi_vals,
            alpha=0.3,
            label=f"[{quantile_lo:.0%}, {quantile_hi:.0%}]",
        )

    def plot_timeseries_at_station(
        self,
        ax: Any,
        station_idx: int,
        var: str,
        time_indices: list[int],
        quantile_lo: float = 0.05,
        quantile_hi: float = 0.95,
    ) -> None:
        """Plot predicted timeseries at a station location, with observed values.

        Wraps :meth:`plot_timeseries` and additionally overlays the actual
        station observations as a scatter plot.  Missing observations
        (``obs_mask_tensor == 0``) are rendered as NaN and therefore omitted.

        Args:
            ax: Matplotlib Axes to draw onto.
            station_idx: Index into ``dataset.station_coords``.
            var: Variable name (must be in predictor.var_to_indx).
            time_indices: Dataset time indices to predict at.
            quantile_lo: Lower quantile for the uncertainty band.
            quantile_hi: Upper quantile for the uncertainty band.
        """
        coords = self.dataset.station_coords[station_idx]
        center = (coords[0].item(), coords[1].item())
        self.plot_timeseries(
            ax,
            center,
            var,
            time_indices,
            quantile_lo=quantile_lo,
            quantile_hi=quantile_hi,
        )

        var_idx = self.predictor.var_to_indx[var]
        times = [self.dataset.times[i] for i in time_indices]
        obs_values = []
        for t_idx in time_indices:
            mask = self.dataset.obs_mask_tensor[t_idx, station_idx, var_idx].item()
            if mask:
                obs_values.append(
                    self.dataset.obs_tensor[t_idx, station_idx, var_idx].item()
                )
            else:
                obs_values.append(float("nan"))
        ax.scatter(times, obs_values, label="observation", zorder=5, c="k")

    def plot_spatial_snapshot(
        self,
        ax: Any,
        var: str,
        time_idx: int,
        center: tuple[float, float],
        extent: float = 5.0,
        resolution: float = 0.1,
        plot_type: Literal["mean", "std"] = "mean",
        show_stations: bool = True,
    ) -> Any:
        """Plot a spatial field at a single timestep.

        Creates a regular grid centered at *center*, runs inference on every
        grid cell, and renders a pcolormesh of either the predictive mean or
        std for *var* onto *ax*.

        Args:
            ax: Matplotlib Axes to draw onto.
            var: Variable name (must be in predictor.var_to_indx).
            time_idx: Dataset time index to predict at.
            center: Grid center as (x, y) in the dataset's coordinate frame.
            extent: Half-extent of the grid in coordinate units (default 5.0).
            resolution: Grid spacing in coordinate units (default 0.1).
            plot_type: "mean" or "std" of the predictive distribution.
            show_stations: If True, overlay all dataset stations as scatter
                markers, distinguishing observed (in context, mask==1) from
                not observed.

        Returns:
            The QuadMesh returned by ``ax.pcolormesh``.

        Raises:
            ValueError: If plot_type is not "mean" or "std".
        """
        if plot_type not in ("mean", "std"):
            raise ValueError(f"plot_type must be 'mean' or 'std', got {plot_type!r}.")

        var_idx = self.predictor.var_to_indx[var]

        xs = np.arange(center[0] - extent, center[0] + extent + 1e-9, resolution)
        ys = np.arange(center[1] - extent, center[1] + extent + 1e-9, resolution)
        x_grid, y_grid = np.meshgrid(xs, ys)
        query_coords = torch.tensor(
            np.stack([x_grid.ravel(), y_grid.ravel()], axis=1),
            dtype=torch.float32,
        )  # [N_q, 2]

        # [N_q, n_samples, n_vars]
        samples = self._run_inference(time_idx, query_coords)
        var_samples = samples[:, :, var_idx]  # [N_q, n_samples]

        if plot_type == "mean":
            field = var_samples.mean(dim=1).numpy()
        else:
            field = var_samples.std(dim=1).numpy()

        field = field.reshape(y_grid.shape)

        mesh = ax.pcolormesh(xs, ys, field, shading="auto")

        ax.set_xlim(center[0] - extent, center[0] + extent)
        ax.set_ylim(center[1] - extent, center[1] + extent)

        if show_stations:
            obs_set = set(self.obs_indices)
            eval_set = set(self.eval_indices)
            n_st = self.dataset.station_coords.shape[0]
            obs_x, obs_y, obs_vals = [], [], []
            missing_obs_x, missing_obs_y = [], []
            sq_x, sq_y = [], []

            for s in range(n_st):
                sx = self.dataset.station_coords[s, 0].item()
                sy = self.dataset.station_coords[s, 1].item()
                mask_val = self.dataset.obs_mask_tensor[time_idx, s, var_idx].item()
                if s in obs_set:
                    if mask_val:
                        obs_x.append(sx)
                        obs_y.append(sy)
                        obs_vals.append(
                            self.dataset.obs_tensor[time_idx, s, var_idx].item()
                        )
                    elif eval_set:
                        missing_obs_x.append(sx)
                        missing_obs_y.append(sy)
                    else:
                        sq_x.append(sx)
                        sq_y.append(sy)
                elif s in eval_set:
                    sq_x.append(sx)
                    sq_y.append(sy)
                elif not eval_set:
                    sq_x.append(sx)
                    sq_y.append(sy)

            if plot_type == "mean" and obs_vals:
                obs_colors = mesh.cmap(mesh.norm(np.array(obs_vals)))
            else:
                obs_colors = "white"

            ax.scatter(
                obs_x,
                obs_y,
                marker="o",
                c=obs_colors,
                edgecolors="black",
                zorder=5,
                label="observed",
            )
            if missing_obs_x:
                ax.scatter(
                    missing_obs_x,
                    missing_obs_y,
                    marker="o",
                    facecolors="white",
                    edgecolors="black",
                    zorder=5,
                    label="missing",
                )
            sq_label = "eval" if eval_set else "not observed"
            ax.scatter(
                sq_x,
                sq_y,
                marker="s",
                facecolors="white",
                edgecolors="black",
                zorder=5,
                label=sq_label,
            )
            ax.legend()
            ax.scatter(
                center[0], center[1], marker="X", c="red", s=100, zorder=6, label="POI"
            )

        return mesh

    def plot_spatial_snapshot_at_station(
        self,
        ax: Any,
        var: str,
        time_idx: int,
        station_idx: int,
        extent: float = 5.0,
        resolution: float = 0.1,
        plot_type: Literal["mean", "std"] = "mean",
        show_stations: bool = True,
    ) -> Any:
        """Plot a spatial snapshot centered on a station location.

        Wraps :meth:`plot_spatial_snapshot`, resolving the grid center from
        ``dataset.station_coords[station_idx]``.

        Args:
            ax: Matplotlib Axes to draw onto.
            var: Variable name (must be in predictor.var_to_indx).
            time_idx: Dataset time index to predict at.
            station_idx: Index into ``dataset.station_coords``.
            extent: Half-extent of the grid in coordinate units (default 5.0).
            resolution: Grid spacing in coordinate units (default 0.1).
            plot_type: "mean" or "std" of the predictive distribution.
            show_stations: Passed through to :meth:`plot_spatial_snapshot`.

        Returns:
            The QuadMesh returned by ``ax.pcolormesh``.
        """
        coords = self.dataset.station_coords[station_idx]
        center = (coords[0].item(), coords[1].item())
        return self.plot_spatial_snapshot(
            ax,
            var,
            time_idx,
            center,
            extent=extent,
            resolution=resolution,
            plot_type=plot_type,
            show_stations=show_stations,
        )
