from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch

from oat.data_module.data_split import _slice_field
from oat.data_module.field import Field
from oat.data_module.setup import collate

if TYPE_CHECKING:
    from oat.predictor_module.base_predictor import BasePredictor


def _field_to_device(field: Field, device: torch.device) -> Field:
    """Return a copy of *field* with its tensors moved to *device*."""
    mask = field.mask.to(device) if field.mask is not None else None
    return Field(
        field.coords.to(device), field.values.to(device), list(field.var_names), mask
    )


class OATPlotter:
    """Wraps a BasePredictor and a dataset to produce diagnostic plots.

    Args:
        predictor: Any BasePredictor implementation (e.g. OATPredictor or
            NWPInterpolatorPredictor).
        dataset: Field-path `DatasetBase` instance with `station_coords`,
            `obs_tensor`, `obs_mask_tensor`, `times`, `roles()`, `__getitem__`
            returning `{"index", "fields"}`, and `get_query(coords, idx)`
            returning `dict[str, Field]`.
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
        predictor: BasePredictor,
        dataset: Any,
        obs_station_indices: Sequence[int] | None = None,
        eval_indices: Sequence[int] | None = None,
        n_samples: int = 100,
        device: str = "cpu",
        plot_config: Any = None,
    ) -> None:
        self.predictor = predictor
        self.dataset = dataset
        if obs_station_indices is None:
            self.obs_indices: list[int] = list(range(dataset.station_coords.shape[0]))
        else:
            self.obs_indices = [int(i) for i in obs_station_indices]
        if eval_indices is None:
            self.eval_indices: list[int] = []
        else:
            self.eval_indices = [int(i) for i in eval_indices]
        self.n_samples = n_samples
        self.device = torch.device(device)
        self._plot_config = plot_config

    def _get_var_plot_config(self, var: str) -> Any:
        raw = self._plot_config
        if raw is None:
            try:
                raw = self.dataset.config.obs_plot_config
            except AttributeError:
                return {}
        if raw is None or var not in raw:
            return {}
        return raw[var]

    def _run_inference(
        self,
        time_idx: int,
        query_coords: torch.Tensor,
        predictor: BasePredictor | None = None,
        stage: str = "oat",
    ) -> torch.Tensor:
        """Run one forward pass and return unnormalized samples.

        Args:
            time_idx: Index into `dataset.times`.
            query_coords: Shape [N_q, 2] in the dataset's coordinate frame.
            predictor: If provided, use this predictor instead of self.predictor
                for normalization and sampling. The dataset and indices from
                self are always used.
            stage: "oat" for the full prediction, "pre_oat" for the model's
                output before the assimilation block (OAT predictor only).
                Only forwarded to `predict_step` when not "oat", since
                baseline predictors' `predict_step` has no `stage` parameter.

        Returns:
            Samples tensor of shape [N_q, n_samples, n_vars] (unnormalized).
        """
        p = predictor if predictor is not None else self.predictor
        p = p.to(self.device)

        roles = self.dataset.roles()
        target_name = roles["target"]

        all_fields = self.dataset[time_idx]["fields"]
        obs_idx = torch.as_tensor(self.obs_indices, dtype=torch.long)
        obs_fields = {n: _slice_field(f, obs_idx) for n, f in all_fields.items()}

        query_fields = self.dataset.get_query(query_coords, time_idx)
        coords = next(iter(query_fields.values())).coords  # [N_q, 3]: (x, y, t)
        n_obs_vars = len(all_fields[target_name].var_names)
        query_fields[target_name] = Field(
            coords,
            torch.zeros(coords.shape[0], n_obs_vars),
            list(all_fields[target_name].var_names),
            torch.zeros(coords.shape[0], n_obs_vars),
        )

        sample = {
            "index": time_idx,
            "observation_data": {"fields": obs_fields},
            "query_data": {"fields": query_fields},
        }
        batch = collate([sample])
        for group in ("observation_data", "query_data"):
            batch[group]["fields"] = {
                n: _field_to_device(f, self.device)
                for n, f in batch[group]["fields"].items()
            }

        if stage == "oat":
            samples = p.predict_step(batch, batch_idx=0, n_samples=self.n_samples)
        else:
            samples = p.predict_step(
                batch, batch_idx=0, n_samples=self.n_samples, stage=stage
            )
        # samples: [1, N_q, n_samples, n_vars]
        return samples[0]  # [N_q, n_samples, n_vars]

    def predict_samples(
        self,
        query_coords: torch.Tensor,
        time_idx: int,
        predictor: BasePredictor | None = None,
        stage: str = "oat",
    ) -> torch.Tensor:
        """Public wrapper over :meth:`_run_inference`.

        Args:
            query_coords: [N_q, 2] coordinates in the dataset frame.
            time_idx: Index into ``dataset.times``.
            predictor: Optional predictor override (defaults to self.predictor).
            stage: "oat" for the full prediction, "pre_oat" for the model's
                output before the assimilation block (OAT predictor only).

        Returns:
            Unnormalized samples of shape [N_q, n_samples, n_vars].
        """
        return self._run_inference(
            time_idx, query_coords, predictor=predictor, stage=stage
        )

    def overlay_stations(
        self,
        ax: Any,
        time_idx: int,
        var_idx: int,
        mesh: Any = None,
        color_by: dict[int, float] | None = None,
    ) -> None:
        """Public wrapper around :meth:`_overlay_stations`.

        Observed context stations are drawn as circles and evaluation stations
        as squares, coloured by observed value (or by *color_by*) when *mesh*
        is given. See :meth:`_overlay_stations` for details.
        """
        self._overlay_stations(ax, time_idx, var_idx, mesh=mesh, color_by=color_by)

    def _overlay_stations(
        self,
        ax: Any,
        time_idx: int,
        var_idx: int,
        mesh: Any = None,
        color_by: dict[int, float] | None = None,
    ) -> None:
        """Overlay station markers on *ax*, optionally colour-coded by a mesh.

        Observed stations (in self.obs_indices with mask==1) are drawn as circles;
        evaluation stations (self.eval_indices) as squares.  Pass *mesh* to derive
        colours from its colormap/norm; pass *color_by* (station_idx → scalar) to
        override the per-station value used for colouring (e.g. obs − NWP).
        """
        if color_by is not None:
            # Normalize keys to plain int: a caller may build color_by from a
            # 0-dim-Tensor-valued station index (e.g. list(tensor)), which
            # would otherwise silently never match `s` below (int keys/hash).
            color_by = {int(k): v for k, v in color_by.items()}
        obs_set = set(self.obs_indices)
        eval_set = set(self.eval_indices)
        n_st = self.dataset.station_coords.shape[0]
        obs_x: list[float] = []
        obs_y: list[float] = []
        obs_vals: list[float] = []
        missing_obs_x: list[float] = []
        missing_obs_y: list[float] = []
        eval_x: list[float] = []
        eval_y: list[float] = []
        eval_vals: list[float] = []
        missing_eval_x: list[float] = []
        missing_eval_y: list[float] = []
        sq_x: list[float] = []
        sq_y: list[float] = []

        for s in range(n_st):
            sx = self.dataset.station_coords[s, 0].item()
            sy = self.dataset.station_coords[s, 1].item()
            mask_val = self.dataset.obs_mask_tensor[time_idx, s, var_idx].item()
            if s in obs_set:
                if mask_val:
                    obs_x.append(sx)
                    obs_y.append(sy)
                    val = (
                        color_by[s]
                        if (color_by is not None and s in color_by)
                        else self.dataset.obs_tensor[time_idx, s, var_idx].item()
                    )
                    obs_vals.append(val)
                elif eval_set:
                    missing_obs_x.append(sx)
                    missing_obs_y.append(sy)
                else:
                    sq_x.append(sx)
                    sq_y.append(sy)
            elif s in eval_set:
                if mask_val:
                    eval_x.append(sx)
                    eval_y.append(sy)
                    val = (
                        color_by[s]
                        if (color_by is not None and s in color_by)
                        else self.dataset.obs_tensor[time_idx, s, var_idx].item()
                    )
                    eval_vals.append(val)
                else:
                    missing_eval_x.append(sx)
                    missing_eval_y.append(sy)
            elif not eval_set:
                sq_x.append(sx)
                sq_y.append(sy)

        if mesh is not None and obs_vals:
            obs_colors: Any = mesh.cmap(mesh.norm(np.array(obs_vals)))
            obs_label = "obs − NWP" if color_by is not None else "observed"
        else:
            obs_colors = "white"
            obs_label = "observed"

        if mesh is not None and eval_vals:
            eval_colors: Any = mesh.cmap(mesh.norm(np.array(eval_vals)))
            eval_label = "eval obs − NWP" if color_by is not None else "eval observed"
        else:
            eval_colors = "white"
            eval_label = "eval observed"

        if obs_x:
            ax.scatter(
                obs_x,
                obs_y,
                marker="o",
                c=obs_colors,
                edgecolors="black",
                s=10,
                zorder=5,
                label=obs_label,
            )
        if missing_obs_x:
            ax.scatter(
                missing_obs_x,
                missing_obs_y,
                marker="o",
                facecolors="white",
                edgecolors="grey",
                s=10,
                zorder=5,
                label="missing",
            )
        if eval_x:
            ax.scatter(
                eval_x,
                eval_y,
                marker="s",
                c=eval_colors,
                edgecolors="black",
                s=14,
                zorder=5,
                label=eval_label,
            )
        if missing_eval_x:
            ax.scatter(
                missing_eval_x,
                missing_eval_y,
                marker="s",
                facecolors="white",
                edgecolors="grey",
                s=14,
                zorder=5,
                label="eval missing",
            )
        if sq_x:
            ax.scatter(
                sq_x,
                sq_y,
                marker="s",
                facecolors="white",
                edgecolors="black",
                s=10,
                zorder=5,
                label="not observed",
            )

    def plot_timeseries(
        self,
        ax: Any,
        center: tuple[float, float],
        var: str,
        time_indices: list[int],
        quantile_lo: float = 0.05,
        quantile_hi: float = 0.95,
        predictors: dict[str, BasePredictor] | None = None,
    ) -> None:
        """Plot predicted distribution over time at an arbitrary location.

        Draws a mean line and a shaded uncertainty band onto *ax*.  If
        *predictors* is given, each named predictor is additionally plotted as
        a point prediction (median) line without an uncertainty band.

        Args:
            ax: Matplotlib Axes to draw onto.
            center: Query location as (x, y) in the dataset's coordinate frame.
            var: Name of the variable to plot (must be in predictor.var_to_indx).
            time_indices: Dataset time indices to predict at.
            quantile_lo: Lower quantile for the uncertainty band.
            quantile_hi: Upper quantile for the uncertainty band.
            predictors: Optional dict mapping display name → predictor.  Each
                entry is plotted as a single median line (no band).
        """
        var_idx = self.predictor.var_to_indx[var]
        var_cfg = self._get_var_plot_config(var)
        query_coords = torch.tensor([center], dtype=torch.float32)  # [1, 2]

        medians, lo_vals, hi_vals = [], [], []

        for t_idx in time_indices:
            samples = self._run_inference(t_idx, query_coords)  # [1, n_samples, n_vars]
            var_samples = samples[0, :, var_idx]  # [n_samples]
            medians.append(var_samples.median().item())
            lo_vals.append(var_samples.quantile(quantile_lo).item())
            hi_vals.append(var_samples.quantile(quantile_hi).item())

        times = [self.dataset.times[i] for i in time_indices]

        ax.plot(times, medians, label="median", color=var_cfg.get("color", "C0"))
        ax.fill_between(
            times,
            lo_vals,
            hi_vals,
            alpha=0.3,
            label=f"[{quantile_lo:.0%}, {quantile_hi:.0%}]",
        )

        if predictors:
            for name, pred in predictors.items():
                pred_medians = []
                for t_idx in time_indices:
                    samples = self._run_inference(t_idx, query_coords, predictor=pred)
                    var_samples = samples[0, :, var_idx]
                    pred_medians.append(var_samples.median().item())
                ax.plot(times, pred_medians, label=name)

    def plot_timeseries_at_station(
        self,
        ax: Any,
        station_idx: int,
        var: str,
        time_indices: list[int],
        quantile_lo: float = 0.05,
        quantile_hi: float = 0.95,
        predictors: dict[str, BasePredictor] | None = None,
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
            predictors: Optional dict mapping display name → predictor.  Each
                entry is plotted as a single median line (no band).
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
            predictors=predictors,
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
        var_cfg = self._get_var_plot_config(var)

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
            field = var_samples.mean(dim=1).detach().cpu().numpy()
        else:
            field = var_samples.std(dim=1).detach().cpu().numpy()

        field = field.reshape(y_grid.shape)

        cmap = var_cfg.get("cmap", "viridis") if plot_type == "mean" else None
        vmin = var_cfg.get("vmin") if plot_type == "mean" else None
        vmax = var_cfg.get("vmax") if plot_type == "mean" else None
        mesh = ax.pcolormesh(xs, ys, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")

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

            if obs_x:
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
            if sq_x:
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

    def plot_spatial_difference(
        self,
        other: BasePredictor,
        ax: Any,
        var: str,
        time_idx: int,
        center: tuple[float, float],
        extent: float = 5.0,
        resolution: float = 0.1,
        show_stations: bool = True,
    ) -> Any:
        """Plot the difference of predicted mean spatial fields for two predictors.

        Creates a regular grid centered at *center*, runs inference with both
        ``self.predictor`` and *other*, and renders a diverging pcolormesh of
        the mean difference (self − other) for *var* onto *ax*.  The dataset,
        observation indices, number of samples, and device are all taken from
        ``self``.

        Args:
            other: Second predictor whose mean is subtracted from self's mean.
            ax: Matplotlib Axes to draw onto.
            var: Variable name (must be in predictor.var_to_indx).
            time_idx: Dataset time index to predict at.
            center: Grid center as (x, y) in the dataset's coordinate frame.
            extent: Half-extent of the grid in coordinate units (default 5.0).
            resolution: Grid spacing in coordinate units (default 0.1).
            show_stations: If True, overlay dataset stations as scatter markers.
                Observed station markers are color-coded by (obs − NWP predicted
                mean), using the same diverging colormap as the background field.

        Returns:
            The QuadMesh returned by ``ax.pcolormesh``.
        """
        var_idx = self.predictor.var_to_indx[var]

        xs = np.arange(center[0] - extent, center[0] + extent + 1e-9, resolution)
        ys = np.arange(center[1] - extent, center[1] + extent + 1e-9, resolution)
        x_grid, y_grid = np.meshgrid(xs, ys)
        query_coords = torch.tensor(
            np.stack([x_grid.ravel(), y_grid.ravel()], axis=1),
            dtype=torch.float32,
        )  # [N_q, 2]

        samples_a = self._run_inference(time_idx, query_coords)
        samples_b = self._run_inference(time_idx, query_coords, predictor=other)

        diff = (
            samples_a[:, :, var_idx].mean(dim=1) - samples_b[:, :, var_idx].mean(dim=1)
        ).detach().cpu().numpy()
        field = diff.reshape(y_grid.shape)

        vmax = float(np.abs(field).max()) or 1.0
        mesh = ax.pcolormesh(
            xs, ys, field, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
        )

        ax.set_xlim(center[0] - extent, center[0] + extent)
        ax.set_ylim(center[1] - extent, center[1] + extent)

        if show_stations:
            obs_set = set(self.obs_indices)
            eval_set = set(self.eval_indices)
            n_st = self.dataset.station_coords.shape[0]
            obs_x, obs_y, obs_vals, obs_coords = [], [], [], []
            eval_x, eval_y, eval_vals, eval_coords = [], [], [], []
            missing_obs_x, missing_obs_y = [], []
            eval_missing_x, eval_missing_y = [], []
            unobs_x, unobs_y = [], []

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
                        obs_coords.append([sx, sy])
                    elif eval_set:
                        missing_obs_x.append(sx)
                        missing_obs_y.append(sy)
                    else:
                        unobs_x.append(sx)
                        unobs_y.append(sy)
                elif s in eval_set:
                    if mask_val:
                        eval_x.append(sx)
                        eval_y.append(sy)
                        eval_vals.append(
                            self.dataset.obs_tensor[time_idx, s, var_idx].item()
                        )
                        eval_coords.append([sx, sy])
                    else:
                        eval_missing_x.append(sx)
                        eval_missing_y.append(sy)
                elif not eval_set:
                    unobs_x.append(sx)
                    unobs_y.append(sy)

            # Run NWP inference for all colorable stations (obs + eval) in one call.
            all_colored_coords = obs_coords + eval_coords
            if all_colored_coords:
                station_query = torch.tensor(all_colored_coords, dtype=torch.float32)
                nwp_at_stations = self._run_inference(
                    time_idx, station_query, predictor=other
                )
                nwp_means = nwp_at_stations[:, :, var_idx].mean(dim=1).numpy()
                all_diffs = np.array(obs_vals + eval_vals) - nwp_means
                n_obs = len(obs_coords)
                colors = mesh.cmap(mesh.norm(all_diffs))
                if obs_x:
                    ax.scatter(
                        obs_x,
                        obs_y,
                        marker="o",
                        c=colors[:n_obs],
                        edgecolors="black",
                        zorder=5,
                        label="obs − NWP",
                    )
                if eval_x:
                    ax.scatter(
                        eval_x,
                        eval_y,
                        marker="s",
                        c=colors[n_obs:],
                        edgecolors="black",
                        zorder=5,
                        label="eval − NWP",
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
            if eval_missing_x:
                ax.scatter(
                    eval_missing_x,
                    eval_missing_y,
                    marker="s",
                    facecolors="white",
                    edgecolors="black",
                    zorder=5,
                    label="eval missing",
                )
            if unobs_x:
                ax.scatter(
                    unobs_x,
                    unobs_y,
                    marker="s",
                    facecolors="white",
                    edgecolors="black",
                    zorder=5,
                    label="not observed",
                )
            ax.legend()
            ax.scatter(
                center[0], center[1], marker="X", c="red", s=100, zorder=6, label="POI"
            )

        return mesh

    def plot_spatial_difference_at_station(
        self,
        other: BasePredictor,
        ax: Any,
        var: str,
        time_idx: int,
        station_idx: int,
        extent: float = 5.0,
        resolution: float = 0.1,
        show_stations: bool = True,
    ) -> Any:
        """Plot a spatial difference centered on a station location.

        Wraps :meth:`plot_spatial_difference`, resolving the grid center from
        ``dataset.station_coords[station_idx]``.

        Args:
            other: Second predictor whose mean is subtracted from self's mean.
            ax: Matplotlib Axes to draw onto.
            var: Variable name (must be in predictor.var_to_indx).
            time_idx: Dataset time index to predict at.
            station_idx: Index into ``dataset.station_coords``.
            extent: Half-extent of the grid in coordinate units (default 5.0).
            resolution: Grid spacing in coordinate units (default 0.1).
            show_stations: Passed through to :meth:`plot_spatial_difference`.

        Returns:
            The QuadMesh returned by ``ax.pcolormesh``.
        """
        coords = self.dataset.station_coords[station_idx]
        center = (coords[0].item(), coords[1].item())
        return self.plot_spatial_difference(
            other,
            ax,
            var,
            time_idx,
            center,
            extent=extent,
            resolution=resolution,
            show_stations=show_stations,
        )

    def compute_station_metrics(
        self,
        time_indices: list[int],
        eval_station_indices: list[int],
        reference: BasePredictor | None = None,
        batch_size: int = 32,
    ) -> dict[str, torch.Tensor]:
        """Compute CRPS and MAE (and optionally CRPSS) over eval stations.

        Args:
            time_indices: Dataset time indices to evaluate over.
            eval_station_indices: Station indices to treat as query targets.
            reference: Optional baseline predictor for CRPS Skill Score.
            batch_size: Number of timesteps processed per forward pass to bound
                peak memory (default 32).

        Returns:
            Dict with keys "crps", "mae", "crps_global", "mae_global", and
            optionally "crpss", "crpss_global" when *reference* is given.
            Shapes: per-station tensors are [n_eval_stations, n_vars]; globals are scalars.
        """
        import logging
        from itertools import batched

        from oat.predictor_module.functional_metrics import mc_crps, point_mae

        logger = logging.getLogger(__name__)

        eval_coords = torch.stack(
            [self.dataset.station_coords[s] for s in eval_station_indices]
        )  # [S, 2]

        S = len(eval_station_indices)
        V = self.dataset.obs_tensor.shape[-1]
        crps_sum = torch.zeros(S, V, device=self.device)
        mae_sum = torch.zeros(S, V, device=self.device)
        mask_count = torch.zeros(S, V, device=self.device)
        ref_crps_sum = (
            torch.zeros(S, V, device=self.device) if reference is not None else None
        )

        processed = 0
        total = len(time_indices)
        for chunk in batched(time_indices, batch_size):
            chunk_list = list(chunk)
            y_hat_b = torch.stack(
                [self._run_inference(t, eval_coords) for t in chunk_list]
            )  # [B, S, n_samples, V]
            y_obs_b = torch.stack(
                [
                    torch.stack([self.dataset.obs_tensor[t, s] for s in eval_station_indices])
                    for t in chunk_list
                ]
            ).to(self.device)  # [B, S, V]
            mask_b = torch.stack(
                [
                    torch.stack(
                        [self.dataset.obs_mask_tensor[t, s] for s in eval_station_indices]
                    )
                    for t in chunk_list
                ]
            ).float().to(self.device)  # [B, S, V]

            count_b = mask_b.sum(dim=0)  # [S, V]
            crps_sum += mc_crps(y_obs_b, y_hat_b, mask_b, aggregation="station") * count_b
            mae_sum += point_mae(y_obs_b, y_hat_b, mask_b, aggregation="station") * count_b
            mask_count += count_b

            if reference is not None:
                y_hat_ref_b = torch.stack(
                    [self._run_inference(t, eval_coords, predictor=reference)
                     for t in chunk_list]
                )
                ref_crps_sum += mc_crps(y_obs_b, y_hat_ref_b, mask_b, aggregation="station") * count_b  # type: ignore[operator]

            processed += len(chunk_list)
            logger.info("processed %d/%d timesteps", processed, total)

        denom = mask_count.clamp(min=1e-8)
        no_obs = mask_count == 0  # [S, V] — station has no valid obs for this variable
        crps_per_station = (crps_sum / denom).masked_fill(no_obs, float("nan"))  # [S, V]
        mae_per_station = (mae_sum / denom).masked_fill(no_obs, float("nan"))    # [S, V]
        crps_global = crps_sum.sum() / mask_count.sum().clamp(min=1e-8)
        mae_global = mae_sum.sum() / mask_count.sum().clamp(min=1e-8)

        result: dict[str, torch.Tensor] = {
            "crps": crps_per_station,
            "mae": mae_per_station,
            "crps_global": crps_global,
            "mae_global": mae_global,
        }

        if reference is not None:
            ref_crps_per_station = (ref_crps_sum / denom).masked_fill(no_obs, float("nan"))  # type: ignore[operator]
            ref_crps_global = ref_crps_sum.sum() / mask_count.sum().clamp(min=1e-8)  # type: ignore[union-attr]
            crpss_per_station = 1.0 - crps_per_station / ref_crps_per_station.clamp(min=1e-8)
            result["crpss"] = crpss_per_station.masked_fill(no_obs, float("nan"))
            result["crpss_global"] = 1.0 - crps_global / ref_crps_global.clamp(min=1e-8)

        return result

    def compute_station_time_metrics(
        self,
        time_indices: list[int],
        eval_station_indices: list[int],
        batch_size: int = 32,
    ) -> dict[str, torch.Tensor]:
        """Per-(time, station, variable) CRPS and MAE (no time aggregation).

        Args:
            time_indices: Dataset time indices to evaluate over.
            eval_station_indices: Station indices to treat as query targets.
            batch_size: Timesteps to log progress over (inference is per-step).

        Returns:
            Dict with keys "crps" and "mae", each a tensor of shape
            [T, S, V] with T = len(time_indices). Entries are NaN where the
            observation mask is 0.
        """
        import logging
        from itertools import batched

        from oat.predictor_module.functional_metrics import mc_crps, point_mae

        logger = logging.getLogger(__name__)

        eval_coords = torch.stack(
            [self.dataset.station_coords[s] for s in eval_station_indices]
        )  # [S, 2]
        T = len(time_indices)
        S = len(eval_station_indices)
        V = self.dataset.obs_tensor.shape[-1]
        crps_out = torch.full((T, S, V), float("nan"))
        mae_out = torch.full((T, S, V), float("nan"))

        pos = 0
        for chunk in batched(time_indices, batch_size):
            for t in chunk:
                y_obs = self.dataset.obs_tensor[t, eval_station_indices].unsqueeze(0).to(
                    self.device
                )  # [1, S, V]
                mask = (
                    self.dataset.obs_mask_tensor[t, eval_station_indices]
                    .float()
                    .unsqueeze(0)
                    .to(self.device)
                )  # [1, S, V]
                if mask.sum().item() == 0.0:
                    pos += 1
                    continue
                y_hat = self._run_inference(t, eval_coords).unsqueeze(0)  # [1,S,n,V]
                crps_sv = mc_crps(y_obs, y_hat, mask, aggregation="station")  # [S,V]
                mae_sv = point_mae(y_obs, y_hat, mask, aggregation="station")  # [S,V]
                no_obs = mask[0] == 0
                crps_out[pos] = crps_sv.masked_fill(no_obs, float("nan")).cpu()
                mae_out[pos] = mae_sv.masked_fill(no_obs, float("nan")).cpu()
                pos += 1
            logger.info("processed %d/%d timesteps", pos, T)

        return {"crps": crps_out, "mae": mae_out}

    def plot_domain_overview(
        self,
        other: BasePredictor,
        var: str,
        time_idx: int,
        xs: np.ndarray,
        ys: np.ndarray,
        title: str = "",
        show_stations: bool = True,
    ) -> Any:
        """Produce a 4-panel domain diagnostic figure.

        Panels: (1) NWP interpolation mean, (2) OAT predicted mean,
        (3) OAT predictive std, (4) OAT − NWP difference with obs − NWP
        station colouring.

        Args:
            other: NWP baseline predictor (subtracted from self in panel 4).
            var: Variable name (must be in predictor.var_to_indx).
            time_idx: Dataset time index to predict at.
            xs: 1-D array of x coordinates for the query grid.
            ys: 1-D array of y coordinates for the query grid.
            title: Optional figure suptitle.

        Returns:
            The matplotlib Figure.  The caller is responsible for saving/closing.
        """
        var_idx = self.predictor.var_to_indx[var]
        var_cfg = self._get_var_plot_config(var)
        x_grid, y_grid = np.meshgrid(xs, ys)
        query_coords = torch.tensor(
            np.stack([x_grid.ravel(), y_grid.ravel()], axis=1),
            dtype=torch.float32,
        )

        samples_nwp = self._run_inference(time_idx, query_coords, predictor=other)
        nwp_mean = (
            samples_nwp[:, :, var_idx].mean(dim=1).detach().cpu().numpy().reshape(y_grid.shape)
        )

        samples_oat = self._run_inference(time_idx, query_coords)
        oat_mean = (
            samples_oat[:, :, var_idx].mean(dim=1).detach().cpu().numpy().reshape(y_grid.shape)
        )
        oat_std = (
            samples_oat[:, :, var_idx].std(dim=1).detach().cpu().numpy().reshape(y_grid.shape)
        )

        diff = oat_mean - nwp_mean
        vmax_diff = float(np.abs(diff).max()) or 1.0

        # Compute obs − NWP differences for coloured station markers in panel 4.
        obs_set = set(self.obs_indices)
        eval_set = set(self.eval_indices)
        colored_indices = [
            s
            for s in range(self.dataset.station_coords.shape[0])
            if (s in obs_set or s in eval_set)
            and self.dataset.obs_mask_tensor[time_idx, s, var_idx].item()
        ]
        color_by_diff: dict[int, float] = {}
        if colored_indices:
            st_coords = torch.tensor(
                [
                    [
                        self.dataset.station_coords[s, 0].item(),
                        self.dataset.station_coords[s, 1].item(),
                    ]
                    for s in colored_indices
                ],
                dtype=torch.float32,
            )
            nwp_at_st = self._run_inference(time_idx, st_coords, predictor=other)
            nwp_means = nwp_at_st[:, :, var_idx].mean(dim=1).numpy()
            for i, s in enumerate(colored_indices):
                obs_val = self.dataset.obs_tensor[time_idx, s, var_idx].item()
                color_by_diff[s] = obs_val - nwp_means[i]

        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        if title:
            fig.suptitle(title)

        mesh_nwp = axes[0].pcolormesh(
            xs, ys, nwp_mean,
            cmap=var_cfg.get("cmap", "viridis"),
            vmin=var_cfg.get("vmin"),
            vmax=var_cfg.get("vmax"),
            shading="auto",
        )
        axes[0].set_title("NWP interp.")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        if show_stations:
            self._overlay_stations(axes[0], time_idx, var_idx, mesh=mesh_nwp)
        plt.colorbar(mesh_nwp, ax=axes[0], label=var)

        mesh_mean = axes[1].pcolormesh(
            xs,
            ys,
            oat_mean,
            shading="auto",
            vmin=mesh_nwp.norm.vmin,
            vmax=mesh_nwp.norm.vmax,
            cmap=mesh_nwp.cmap,
        )
        axes[1].set_title("OAT mean")
        axes[1].set_xlabel("X")
        if show_stations:
            self._overlay_stations(axes[1], time_idx, var_idx, mesh=mesh_mean)
        plt.colorbar(mesh_mean, ax=axes[1], label=var)

        mesh_std = axes[2].pcolormesh(xs, ys, oat_std, shading="auto")
        axes[2].set_title("OAT predictive std")
        axes[2].set_xlabel("X")
        if show_stations:
            self._overlay_stations(axes[2], time_idx, var_idx)
        plt.colorbar(mesh_std, ax=axes[2], label=f"{var} std")

        mesh_diff = axes[3].pcolormesh(
            xs,
            ys,
            diff,
            cmap="RdBu_r",
            vmin=-vmax_diff,
            vmax=vmax_diff,
            shading="auto",
        )
        axes[3].set_title("OAT − NWP  |  markers: obs − NWP")
        axes[3].set_xlabel("X")
        if show_stations:
            self._overlay_stations(
                axes[3], time_idx, var_idx, mesh=mesh_diff, color_by=color_by_diff
            )
        plt.colorbar(mesh_diff, ax=axes[3], label=f"Δ{var}")

        fig.tight_layout()
        return fig

    def compute_pit(
        self,
        time_indices: list[int],
        eval_station_indices: list[int] | None = None,
        log_every: int = 50,
    ) -> dict[str, dict[str, list[float]]]:
        """Compute PIT and zero-inflation diagnostics at eval stations.

        Args:
            time_indices: Dataset time indices to evaluate.
            eval_station_indices: Station indices to query.  Defaults to
                ``self.eval_indices``.
            log_every: Log progress every this many timesteps.

        Returns:
            Dict mapping variable name → dict with keys:

            - ``pit``: PIT values for all valid observations.
            - ``gate_pred``: estimated P(X=0) from samples per valid
              (station, timestep). Non-empty only for zero-inflated variables.
            - ``is_zero``: 1.0/0.0 flag — was the observation exactly zero?
              Non-empty only for zero-inflated variables.
            - ``pit_nonzero``: conditional PIT using only non-zero samples, restricted
              to non-zero observations.  Non-empty only for zero-inflated variables.
        """
        import logging

        logger = logging.getLogger(__name__)

        indices = (
            eval_station_indices
            if eval_station_indices is not None
            else self.eval_indices
        )
        if not indices:
            logger.warning("No eval station indices — returning empty PIT dict.")
            return {
                v: {"pit": [], "gate_pred": [], "is_zero": [], "pit_nonzero": []}
                for v in self.predictor.var_to_indx
            }

        try:
            var_to_dist = self.predictor.model.output_head.variable_to_distribution
        except AttributeError:
            var_to_dist = {}
        zi_vars = {v for v, d in var_to_dist.items() if d == "zero_inflated_lognormal"}

        eval_coords = self.dataset.station_coords[indices, :2]  # [S, 2]
        result: dict[str, dict[str, list[float]]] = {
            v: {"pit": [], "gate_pred": [], "is_zero": [], "pit_nonzero": []}
            for v in self.predictor.var_to_indx
        }

        for i, time_idx in enumerate(time_indices):
            if i % log_every == 0:
                logger.info("PIT: timestep %d / %d", i, len(time_indices))

            with torch.no_grad():
                samples = self._run_inference(
                    time_idx, eval_coords
                )  # [S, n_samples, n_vars]

            obs = self.dataset.obs_tensor[time_idx][indices]  # [S, n_vars]
            mask = self.dataset.obs_mask_tensor[time_idx][indices]  # [S, n_vars]
            pit = (samples <= obs.unsqueeze(1)).float().mean(dim=1)  # [S, n_vars]

            for var, var_idx in self.predictor.var_to_indx.items():
                data = result[var]
                samps_v = samples[:, :, var_idx]  # [S, n_samples]
                obs_v = obs[:, var_idx]  # [S]
                mask_v = mask[:, var_idx].bool()  # [S]

                if not mask_v.any():
                    continue

                # "pit" maps to a list of values continusly filled from all valid
                # observations (mask==1) for this variable across all timesteps.
                # from that, a histogram can be plotted.
                data["pit"].extend(pit[mask_v, var_idx].cpu().tolist())

                if var in zi_vars:
                    # gate_pred and is_zero accumulate one scalar per valid
                    # (station, timestep) pair.
                    # gate_pred[i] = fraction of samples that were zero (≈ P(dry),
                    # an estimate of the gate); is_zero[i] = 1 if obs==0.
                    # Binning gate_pred vs mean(is_zero) per bin gives the
                    # reliability diagram.
                    data["gate_pred"].extend(
                        (samps_v == 0).float().mean(dim=1)[mask_v].cpu().tolist()
                    )
                    data["is_zero"].extend((obs_v == 0).float()[mask_v].cpu().tolist())

                    # Conditional PIT: fraction of non-zero samples ≤ obs, for
                    # non-zero obs only
                    nonzero_obs = mask_v & (obs_v > 0)
                    if nonzero_obs.any():
                        samps_pos = samps_v > 0
                        n_pos = samps_pos.float().sum(dim=1)
                        n_pos_le = (
                            (samps_pos & (samps_v <= obs_v.unsqueeze(1)))
                            .float()
                            .sum(dim=1)
                        )
                        valid_cond = nonzero_obs & (n_pos > 0)
                        if valid_cond.any():
                            data["pit_nonzero"].extend(
                                (n_pos_le[valid_cond] / n_pos[valid_cond])
                                .cpu()
                                .tolist()
                            )

        return result

    def plot_pit(
        self,
        ax: Any,
        pit_values: list[float],
        n_bins: int = 20,
    ) -> None:
        """Plot a PIT calibration histogram for a single variable onto ax.

        Args:
            ax: Matplotlib Axes to draw onto.
            pit_values: Precomputed PIT values (e.g. from ``compute_pit``).
                Pass ``data[var]["pit"]`` for the unconditional PIT or
                ``data[var]["pit_nonzero"]`` for the conditional non-zero PIT
                of a zero-inflated variable.
            n_bins: Number of histogram bins (default 20).
        """
        pits = np.array(pit_values)
        n = len(pits)
        if n == 0:
            ax.axis("off")
            return
        ax.hist(
            pits,
            bins=n_bins,
            density=True,
            color="#CFCFCF",
            edgecolor="#4A5568",
            linewidth=0.8,
        )
        ax.axhline(1.0, color="0.3", linestyle="--", linewidth=1.0, label="Uniform")
        ax.set_title(f"PIT  (n={n:,})", fontsize=9)
        ax.set_xlabel("PIT value")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    def plot_zero_reliability(
        self,
        ax: Any,
        gate_pred: list[float],
        is_zero: list[float],
    ) -> None:
        """Plot a reliability diagram for the zero-gate of a ZI variable onto ax.

        Args:
            ax: Matplotlib Axes to draw onto.
            gate_pred: Estimated P(zero) per valid (station, timestep) pair
                (``data[var]["gate_pred"]`` from ``compute_pit``).
            is_zero: Observed zero flag (1.0/0.0) per pair
                (``data[var]["is_zero"]`` from ``compute_pit``).
        """
        gate_pred_arr = np.array(gate_pred)
        is_zero_arr = np.array(is_zero)
        bin_edges = np.linspace(0, 1, 11)
        mean_pred, mean_obs = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=False):
            m = (gate_pred_arr >= lo) & (gate_pred_arr < hi)
            if m.sum() > 5:
                mean_pred.append(float(gate_pred_arr[m].mean()))
                mean_obs.append(float(is_zero_arr[m].mean()))
        ax.plot([0, 1], [0, 1], "--", color="0.3", lw=1.0, label="Perfect")
        if mean_pred:
            ax.plot(
                mean_pred, mean_obs, "o-", color="#4A5568", ms=4, lw=1.2, label="Model"
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Zero probability", fontsize=9)
        ax.set_xlabel("Predicted P(zero)", fontsize=8)
        ax.set_ylabel("Observed freq. of zeros", fontsize=8)
        ax.legend(fontsize=7)

    def draw_field(
        self,
        ax: Any,
        field: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        cmap: Any = None,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> Any:
        """Draw a precomputed (len(ys), len(xs)) field and return the QuadMesh.

        Clamps the axis limits to the grid span. Colormap/limits are applied
        as given (the caller may re-scale later via ``mesh.set_clim``).
        """
        mesh = ax.pcolormesh(
            xs, ys, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto"
        )
        ax.set_xlim(float(xs[0]), float(xs[-1]))
        ax.set_ylim(float(ys[0]), float(ys[-1]))
        return mesh
