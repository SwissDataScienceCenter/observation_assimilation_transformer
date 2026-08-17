"""Shared field-batch helpers used by OATPredictor and GPPredictor."""

from typing import Any

import numpy as np
import pandas as pd
import torch

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


def _unwrap_fields_to_model_batch(
    batch: dict[str, Any], roles: dict[str, str], requires_observation: bool
) -> dict[str, Any]:
    """Unwrap a field-collated batch into the shapes OAT/GP models expect."""

    required = ("target", "global", "geo_bias", "topo")
    missing = [k for k in required if k not in roles]
    if missing:
        raise ValueError(
            f"field roles missing required keys {missing}; got {sorted(roles)}"
        )

    def grp(group: str) -> dict[str, torch.Tensor]:
        f = batch[group]["fields"]
        obs = f[roles["target"]]
        return {
            "coords": obs.spatial_coords,
            "geo": f[roles["geo_bias"]].values,
            "topo": f[roles["topo"]].values,
            "obs": obs.values,
            "mask_obs": obs.mask,
            "global_tokens": f[roles["global"]].values,
        }

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
