"""Shared field-batch helpers used by OATPredictor and GPPredictor."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from tnp.data.base import OOTGBatch

_EPOCH = pd.Timestamp("1970-01-01T00:00:00", tz="UTC")


def _time_embedding_from_epoch_hours(t_hours: torch.Tensor) -> torch.Tensor:
    """Reconstruct the sin/cos day+year time embedding from epoch hours.

    Mirrors ``cosmo_peakweather_dataset._get_time_embedding``.
    """
    embs = []
    for th in t_hours.reshape(-1).tolist():
        dt = _EPOCH + pd.Timedelta(hours=float(th))
        seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        day_angle = 2 * np.pi * seconds / 86400
        doy = dt.dayofyear + seconds / 86400
        year_angle = 2 * np.pi * doy / 365.2425
        embs.append(
            [
                np.sin(day_angle),
                np.cos(day_angle),
                np.sin(year_angle),
                np.cos(year_angle),
            ]
        )
    dtype = t_hours.dtype if t_hours.is_floating_point() else torch.float32
    return torch.tensor(embs, dtype=dtype, device=t_hours.device)


def resolve_index_projection(
    dataset: Any,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Return the dataset's bound ``project_index_coords`` iff it overrides the
    ``DatasetBase`` default; else ``None``.

    Unwraps one level of ``TrainDataset``/``EvalDataset``-style wrapper via the
    ``.dataset`` attribute, mirroring ``validate_output_head_supports``.
    """
    from oat.data_module.dataset import DatasetBase

    ds = getattr(dataset, "dataset", dataset)
    if (
        getattr(type(ds), "project_index_coords", None)
        is DatasetBase.project_index_coords
    ):
        return None
    return ds.project_index_coords


def index_projection_dim(
    projection: Callable[[torch.Tensor], torch.Tensor] | None,
) -> int:
    """Output channel width of ``projection`` (2 when ``projection`` is None)."""
    if projection is None:
        return 2
    return int(projection(torch.zeros(1, 2)).shape[-1])


def assert_geo_first_two_are_coords(data_cfg: Any) -> None:
    """Contract for overriding datasets: the geo-bias field's first two inputs
    must be the spatial coords (coords.x, coords.y) aligned with ``coords``."""
    geo_role = data_cfg.roles["geo_bias"]
    inputs = list(data_cfg.fields[geo_role].inputs)
    if inputs[:2] != ["coords.x", "coords.y"]:
        raise ValueError(
            "project_index_coords override requires the geo-bias field's first "
            f"two inputs to be [coords.x, coords.y]; got {inputs[:2]!r} "
            f"for field role '{geo_role}'."
        )


def _unwrap_fields_to_model_batch(
    batch: dict[str, Any],
    roles: dict[str, str],
    requires_observation: bool,
    project_index_coords: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Unwrap a field-collated batch into the shapes OAT/GP models expect.

    When ``project_index_coords`` is given, each group gains a ``search_coords``
    key (projected coords for neighbour search) and its ``geo`` vector becomes
    ``[projected_xyz, standardized elevation...]``. ``coords`` is never mutated.
    """

    required = ("target", "global", "geo_bias", "topo")
    missing = [k for k in required if k not in roles]
    if missing:
        raise ValueError(
            f"field roles missing required keys {missing}; got {sorted(roles)}"
        )

    def grp(group: str) -> dict[str, torch.Tensor]:
        f = batch[group]["fields"]
        obs = f[roles["target"]]
        coords = obs.spatial_coords
        geo_vals = f[roles["geo_bias"]].values
        out = {
            "coords": coords,
            "geo": geo_vals,
            "topo": f[roles["topo"]].values,
            "obs": obs.values,
            "mask_obs": obs.mask,
            "global_tokens": f[roles["global"]].values,
        }
        if project_index_coords is not None:
            proj = project_index_coords(coords)
            out["search_coords"] = proj
            out["geo"] = torch.cat([proj, geo_vals[..., 2:]], dim=-1)
        return out

    query_obs = batch["query_data"]["fields"][roles["target"]]
    t_hours = query_obs.coords[:, 0, 2]
    time_emb = _time_embedding_from_epoch_hours(t_hours).to(query_obs.values.dtype)
    out: dict[str, Any] = {
        "global_data": {"time_emb": time_emb},
        "query_data": grp("query_data"),
    }
    if requires_observation:
        out["observation_data"] = grp("observation_data")
    return out


def _unwrap_fields_to_ootg_batch(
    batch: dict[str, Any],
    roles: dict[str, str],
    coord_to_latlon: Callable[[torch.Tensor], torch.Tensor],
    grid_shape: tuple[int, int],
    nwp_stride: int,
    ablate_obs: bool = False,
    ablate_grid: bool = False,
    project_index_coords: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> "OOTGBatch":
    """Unwrap a field-collated batch into an OOTGBatch for the gridded-TNP model.

    Reshape the flat grid field to ``[B, H, W, C]``, apply the stride, and
    convert coords to lat/lon. Expect ``B == 1`` (context length varies
    after mask filtering).
    """
    from tnp.data.base import OOTGBatch

    obs_f = batch["observation_data"]["fields"][roles["target"]]
    query_f = batch["query_data"]["fields"][roles["target"]]
    grid_f = batch["global_data"]["fields"][roles["global_grid"]]

    h, w = grid_shape
    s = nwp_stride

    # Station coords: [B, N, 3] -> spatial [B, N, 2].
    obs_xy = obs_f.spatial_coords  # [B, N_obs, 2]
    query_xy = query_f.spatial_coords  # [B, N_query, 2]

    # Drop masked obs rows (per sample; B=1 expected).
    if obs_f.mask is not None:
        valid = obs_f.mask[0].all(dim=-1)  # [N_obs]
        obs_xy_valid = obs_xy[0, valid].unsqueeze(0)
        yc = obs_f.values[0, valid].unsqueeze(0)
    else:
        obs_xy_valid = obs_xy
        yc = obs_f.values

    xc = coord_to_latlon(obs_xy_valid[0]).unsqueeze(0)
    xt = coord_to_latlon(query_xy[0]).unsqueeze(0)
    yt = query_f.values

    # Grid: [B, H*W, C] -> [B, H, W, C], stride, coord_to_latlon.
    grid_coords_2d = grid_f.spatial_coords.reshape(-1, h, w, 2)
    grid_vals_2d = grid_f.values.reshape(-1, h, w, grid_f.values.shape[-1])
    strided_coords = grid_coords_2d[:, ::s, ::s]  # [B, H_s, W_s, 2]
    yc_grid = grid_vals_2d[:, ::s, ::s]  # [B, H_s, W_s, C]
    h_s, w_s = yc_grid.shape[1], yc_grid.shape[2]
    xc_grid = coord_to_latlon(strided_coords[0].reshape(-1, 2)).reshape(1, h_s, w_s, 2)

    # Topo (pointwise at station coords): extract if the role exists.
    topo_field = roles.get("topo")
    if topo_field is not None:
        topo_obs_f = batch["observation_data"]["fields"][topo_field]
        topo_query_f = batch["query_data"]["fields"][topo_field]
        if obs_f.mask is not None:
            topo_c = topo_obs_f.values[0, valid].unsqueeze(0)
        else:
            topo_c = topo_obs_f.values
        topo_t = topo_query_f.values

    # Ablations.
    if ablate_obs:
        xc = xc[:, :0]
        yc = yc[:, :0]
        if topo_field is not None:
            topo_c = topo_c[:, :0]
    if ablate_grid:
        yc_grid = torch.zeros_like(yc_grid)

    x = torch.cat([xc, xt], dim=1)
    y = torch.cat([yc, yt], dim=1)

    result = OOTGBatch(
        x=x,
        y=y,
        xc=xc,
        yc=yc,
        xt=xt,
        yt=yt,
        xc_grid=xc_grid,
        yc_grid=yc_grid,
    )
    if topo_field is not None:
        result.topo_c = topo_c  # type: ignore[attr-defined]
        result.topo_t = topo_t  # type: ignore[attr-defined]
    return result
