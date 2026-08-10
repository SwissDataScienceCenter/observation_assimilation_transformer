"""Config-driven field dataset core: assemble a dict of Fields from injected sources."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from oat.data_module.field import Field
from oat.data_module.field_spec import FieldConfig, FieldSpec, validate_field_config
from oat.data_module.sources import Source, SourceInfo

_EPOCH = pd.Timestamp("1970-01-01T00:00:00", tz="UTC")
_AXIS = {"x": 0, "y": 1, "t": 2}


def EPOCH_HOURS(times: pd.DatetimeIndex) -> Tensor:
    """Hours since 1970-01-01T00:00:00Z for each timestamp (tz-aware or UTC-naive)."""
    t = pd.DatetimeIndex(times)
    if t.tz is None:
        t = t.tz_localize("UTC")
    delta = (t - _EPOCH) / pd.Timedelta(hours=1)
    return torch.tensor(delta.to_numpy(dtype="float64"), dtype=torch.float64)


class FieldDatasetBase(Dataset[dict[str, Any]]):
    """Assembles a dict of :class:`Field` per sample from injected coords/sources.

    Dependency-injected core: callers provide the coordinate tensors per coord
    source, the ``Source`` objects to read from, and the time axis. All field
    assembly (which channels, which coord source, masking, query-only subset)
    is driven by the parsed ``FieldConfig``.
    """

    def __init__(
        self,
        *,
        field_config: FieldConfig,
        coord_source_coords: dict[str, Tensor],
        sources: dict[str, Source],
        times: pd.DatetimeIndex,
        obs_var_support: Any,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        validate_field_config(field_config, {n: s.info for n, s in sources.items()})
        self.field_config = field_config
        self._coords = {k: v.to(dtype) for k, v in coord_source_coords.items()}
        self._source_objs = sources
        self.times = pd.DatetimeIndex(times)
        self.obs_var_support = obs_var_support
        self.dtype = dtype
        self._t_hours = EPOCH_HOURS(self.times).to(dtype)

        # Precompute (source, coord_source) nearest indices used by any field.
        self._index_cache: dict[tuple[str, str], Tensor] = {}
        for field in field_config.fields.values():
            for ref in field.inputs:
                if ref.kind == "source":
                    assert ref.source is not None
                    key = (ref.source, field.coords)
                    if key not in self._index_cache:
                        self._index_cache[key] = sources[ref.source].build_index(
                            self._coords[field.coords]
                        )

    def __len__(self) -> int:
        return len(self.times)

    # -- introspection --
    def sources_info(self) -> dict[str, SourceInfo]:
        return {name: s.info for name, s in self._source_objs.items()}

    def field_coord_sources(self) -> dict[str, str]:
        return {n: f.coords for n, f in self.field_config.fields.items()}

    def roles(self) -> dict[str, str]:
        return dict(self.field_config.roles)

    @property
    def variable_names(self) -> dict[str, dict[str, list[str]]]:
        return {"fields": {n: list(f.out) for n, f in self.field_config.fields.items()}}

    # -- assembly --
    def _coords3(self, spatial: Tensor, t_val: Tensor) -> Tensor:
        t_col = t_val.to(spatial.dtype).expand(spatial.shape[0], 1)
        return torch.cat([spatial, t_col], dim=-1)

    def _assemble_field(
        self,
        field: FieldSpec,
        coords3: Tensor,
        index_by_source: dict[str, Tensor],
        frames: dict[str, Tensor],
        mask_frames: dict[str, Tensor | None],
    ) -> Field:
        channels: list[Tensor] = []
        mask_channels: list[Tensor] = []
        masked_source: str | None = None
        for ref in field.inputs:
            if ref.kind == "coord":
                channels.append(coords3[:, _AXIS[ref.name]])
            else:
                assert ref.source is not None
                src = self._source_objs[ref.source]
                idx = index_by_source[ref.source]
                frame = frames[ref.source]
                channels.append(
                    frame[src.var_names.index(ref.name)][idx].to(self.dtype)
                )
                if src.has_mask:
                    masked_source = ref.source
                    mframe = mask_frames[ref.source]
                    assert mframe is not None
                    mask_channels.append(
                        mframe[src.var_names.index(ref.name)][idx].to(self.dtype)
                    )
        values = torch.stack(channels, dim=-1)
        mask = torch.stack(mask_channels, dim=-1) if masked_source is not None else None
        return Field(coords3, values, list(field.out), mask)

    def _build_fields(
        self,
        coords_by_cs: dict[str, Tensor],
        index_by_field: dict[str, dict[str, Tensor]],
        time_idx: int,
        names: list[str],
    ) -> dict[str, Field]:
        t_val = self._t_hours[time_idx].reshape(1)
        coords3_by_cs = {
            cs: self._coords3(sp, t_val) for cs, sp in coords_by_cs.items()
        }
        # Load each needed source's frame once per assembly call.
        needed: set[str] = {
            ref.source
            for n in names
            for ref in self.field_config.fields[n].inputs
            if ref.kind == "source" and ref.source is not None
        }
        frames = {s: self._source_objs[s].load(time_idx) for s in needed}
        mask_frames = {s: self._source_objs[s].load_mask(time_idx) for s in needed}

        out: dict[str, Field] = {}
        for name in names:
            field = self.field_config.fields[name]
            out[name] = self._assemble_field(
                field,
                coords3_by_cs[field.coords],
                index_by_field[name],
                frames,
                mask_frames,
            )
        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        names = list(self.field_config.fields)
        index_by_field = {
            n: {
                ref.source: self._index_cache[
                    (ref.source, self.field_config.fields[n].coords)
                ]
                for ref in self.field_config.fields[n].inputs
                if ref.kind == "source" and ref.source is not None
            }
            for n in names
        }
        fields = self._build_fields(self._coords, index_by_field, idx, names)
        return {"index": idx, "fields": fields}

    def get_query(self, query_coords: Tensor, idx: int) -> dict[str, Field]:
        query_coords = query_coords.to(self.dtype)
        names = [n for n, f in self.field_config.fields.items() if f.query]
        index_by_field = {
            n: {
                ref.source: self._source_objs[ref.source].build_index(query_coords)
                for ref in self.field_config.fields[n].inputs
                if ref.kind == "source" and ref.source is not None
            }
            for n in names
        }
        coords_by_cs = {self.field_config.fields[n].coords: query_coords for n in names}
        return self._build_fields(coords_by_cs, index_by_field, idx, names)
