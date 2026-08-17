"""Config-driven field dataset core: assemble a dict of Fields from injected sources."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from oat.data_module import statistics
from oat.data_module import transform as T
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


class DatasetBase(Dataset[dict[str, Any]]):
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
        self._all_field_names = list(field_config.fields)
        self._query_field_names = [n for n, f in field_config.fields.items() if f.query]
        # Per-source name -> channel index, to avoid a linear var_names.index()
        # scan on every field assembly.
        self._var_index = {
            name: {var: i for i, var in enumerate(src.var_names)}
            for name, src in sources.items()
        }

        # Precompute (source, coord_source, k) indices used by any field.
        # k=1 -> (idx, None) via build_index; k>1 -> (idx, weights) via
        # build_interp (IDW). None marks the unweighted nearest read.
        self._index_cache: dict[tuple[str, str, int], tuple[Tensor, Tensor | None]] = {}
        for field in field_config.fields.values():
            for ref in field.source_refs:
                key = (ref.source, field.coords, field.k)
                if key not in self._index_cache:
                    self._index_cache[key] = self._build_index_entry(
                        sources[ref.source], self._coords[field.coords], field.k, dtype
                    )

    @staticmethod
    def _build_index_entry(
        source: Source, dst_coords: Tensor, k: int, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor | None]:
        """Nearest-neighbor index (k=1), or IDW index + weights (k>1)."""
        if k == 1:
            return source.build_index(dst_coords), None
        idx, weights = source.build_interp(dst_coords, k)
        return idx, weights.to(dtype)

    def __len__(self) -> int:
        return len(self.times)

    # -- introspection --
    def sources_info(self) -> dict[str, SourceInfo]:
        return {name: s.info for name, s in self._source_objs.items()}

    def field_coord_sources(self) -> dict[str, str]:
        return {n: f.coords for n, f in self.field_config.fields.items()}

    def roles(self) -> dict[str, str]:
        return dict(self.field_config.roles)

    def project_index_coords(self, coords: Tensor) -> Tensor:
        """Map spatial coords into the space where Euclidean ops are meaningful.

        Default: identity — coordinates are already treated as Euclidean.
        Datasets whose coordinates are geographic (lon/lat) override this.

        Args:
            coords: (..., 2) spatial coordinates.
        Returns:
            (..., D) coordinates for neighbor search / geo-bias (D may differ).
        """
        return coords

    @property
    def variable_names(self) -> dict[str, dict[str, list[str]]]:
        return {"fields": {n: list(f.out) for n, f in self.field_config.fields.items()}}

    # -- assembly --
    def _coords3(self, spatial: Tensor, t_val: Tensor) -> Tensor:
        """Append the (broadcast) time value as a third coordinate column."""
        t_col = t_val.to(spatial.dtype).expand(spatial.shape[0], 1)
        return torch.cat([spatial, t_col], dim=-1)

    def _assemble_field(
        self,
        field: FieldSpec,
        coords3: Tensor,
        index_by_source: dict[str, tuple[Tensor, Tensor | None]],
        frames: dict[str, Tensor],
        mask_frames: dict[str, Tensor | None],
    ) -> Field:
        """Stack one field's per-channel values (and mask, if its source has one)."""
        channels: list[Tensor] = []
        mask_channels: list[Tensor] = []
        for ref in field.inputs:
            if ref.kind == "coord":
                channels.append(coords3[:, _AXIS[ref.name]])
                continue
            assert ref.source is not None
            src = self._source_objs[ref.source]
            idx, w = index_by_source[ref.source]
            col_idx = self._var_index[ref.source][ref.name]
            vals = frames[ref.source][col_idx][idx].to(self.dtype)
            channels.append(vals if w is None else (vals * w).sum(dim=-1))
            if src.has_mask:
                mframe = mask_frames[ref.source]
                assert mframe is not None
                mask_channels.append(mframe[col_idx][idx].to(self.dtype))
        values = torch.stack(channels, dim=-1)
        mask = torch.stack(mask_channels, dim=-1) if mask_channels else None
        return Field(coords3, values, list(field.out), mask)

    def _build_fields(
        self,
        coords_by_cs: dict[str, Tensor],
        index_by_field: dict[str, dict[str, tuple[Tensor, Tensor | None]]],
        time_idx: int,
        names: list[str],
    ) -> dict[str, Field]:
        """Assemble the named fields for one time step, loading each source once."""
        t_val = self._t_hours[time_idx].reshape(1)
        coords3_by_cs = {
            cs: self._coords3(sp, t_val) for cs, sp in coords_by_cs.items()
        }
        # Load each needed source's frame once per assembly call.
        needed: set[str] = {
            ref.source for n in names for ref in self.field_config.fields[n].source_refs
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

    def _cached_index_entries(
        self, name: str
    ) -> dict[str, tuple[Tensor, Tensor | None]]:
        """Index entries for a field's source inputs, from the precomputed cache."""
        field = self.field_config.fields[name]
        return {
            ref.source: self._index_cache[(ref.source, field.coords, field.k)]
            for ref in field.source_refs
        }

    def _fresh_index_entries(
        self,
        name: str,
        dst_coords: Tensor,
        cache: dict[tuple[str, int], tuple[Tensor, Tensor | None]],
    ) -> dict[str, tuple[Tensor, Tensor | None]]:
        """Index entries built against ``dst_coords``, memoized per (source, k)."""
        field = self.field_config.fields[name]
        entries: dict[str, tuple[Tensor, Tensor | None]] = {}
        for ref in field.source_refs:
            key = (ref.source, field.k)
            if key not in cache:
                cache[key] = self._build_index_entry(
                    self._source_objs[ref.source], dst_coords, field.k, self.dtype
                )
            entries[ref.source] = cache[key]
        return entries

    def __getitem__(self, idx: int) -> dict[str, Any]:
        names = self._all_field_names
        index_by_field = {n: self._cached_index_entries(n) for n in names}
        fields = self._build_fields(self._coords, index_by_field, idx, names)
        return {"index": idx, "fields": fields}

    def get_query(self, query_coords: Tensor, idx: int) -> dict[str, Field]:
        """Assemble query-only fields at arbitrary coords (not the dataset grid)."""
        query_coords = query_coords.to(self.dtype)
        names = self._query_field_names
        cache: dict[tuple[str, int], tuple[Tensor, Tensor | None]] = {}
        index_by_field = {
            n: self._fresh_index_entries(n, query_coords, cache) for n in names
        }
        coords_by_cs = {self.field_config.fields[n].coords: query_coords for n in names}
        return self._build_fields(coords_by_cs, index_by_field, idx, names)

    # -- transforms --
    def _sample_indices(
        self, num_data_points: int, data_indices: Sequence[int] | None
    ) -> list[int]:
        if data_indices is not None:
            pool = np.asarray(data_indices)
            sel = torch.randperm(len(pool))[:num_data_points].numpy()
            return [int(i) for i in pool[sel]]
        sel = torch.randperm(len(self))[:num_data_points].numpy()
        return [int(i) for i in sel]

    def _obs_field_name(self) -> str:
        return self.field_config.roles["target"]

    def target_coord_source(self) -> str:
        return self.field_config.fields[self._obs_field_name()].coords

    @property
    def nwp_extent(self) -> dict[str, Tensor]:
        name = self.field_config.nwp_coord_source
        if name is None:
            raise ValueError("no coord source of kind 'nwp' to derive nwp_extent from")
        xy = self._coords[name][:, :2]
        return {"min": xy.amin(dim=0), "max": xy.amax(dim=0)}

    @staticmethod
    def _support_mask(n: int, predicate: Callable[[int], bool]) -> list[int]:
        return [i for i in range(n) if predicate(i)]

    def _build_obs_pre(
        self, obs_out: list[str], ttype: str, offset: float
    ) -> T.Transform:
        """Build the obs pre-transform (applied before gathering obs stats)."""
        sup = self.obs_var_support
        n = len(obs_out)
        if ttype == "real":
            log_idx = self._support_mask(
                n,
                lambda i: sup.lower_bounds[i] >= 0.0
                and sup.upper_bounds[i] == float("inf"),
            )
            return T.LogTransform(apply_to=log_idx, feature_dim=-1, offset=offset)
        if ttype == "keep_support":
            # offset STRICTLY-positive (open-lower-bound) vars away from zero
            zero_lb = self._support_mask(
                n,
                lambda i: sup.lower_bounds[i] == 0.0
                and bool(sup.lower_open[i])
                and sup.upper_bounds[i] == float("inf"),
            )
            off = torch.zeros(n)
            off[zero_lb] = offset
            return T.TranslateScaleTransform(off, torch.ones(n), channel_dim=-1)
        raise ValueError(f"Unsupported transform_type {ttype!r}")

    def _compose_output_transform(
        self,
        obs_pre: T.Transform,
        obs_stats: statistics.RunningFeatureStats,
        obs_out: list[str],
        ttype: str,
    ) -> T.Transform:
        """Compose obs transform (+ output_transform) from obs_pre + gathered stats."""
        if ttype == "real":
            return T.ComposedTransform([obs_pre, obs_stats.get_standard_transform()])
        sup = self.obs_var_support
        n = len(obs_out)
        translate = -obs_stats.mean.clone()
        only_scale = self._support_mask(
            n,
            lambda i: sup.lower_bounds[i] == 0.0
            and sup.upper_bounds[i] == float("inf"),
        )
        translate[only_scale] = 0.0
        scale = 1.0 / obs_stats.std.clone()
        return T.ComposedTransform(
            [obs_pre, T.TranslateScaleTransform(translate, scale, channel_dim=-1)]
        )

    def compute_field_leaves(
        self,
        num_data_points: int,
        data_indices: Sequence[int] | None = None,
        target_point_indices: Sequence[int] | None = None,
    ) -> tuple[dict[str, T.Transform], T.Transform]:
        """Gather per-field running stats and compose the obs output transform.

        Samples ``num_data_points`` items and gathers per-field running
        statistics over ``Field.values``. The obs field (``roles["target"]``)
        additionally gets a support-aware pre-transform — log (``real``) or a
        zero-avoiding offset (``keep_support``) — applied before its stats are
        gathered.

        If ``target_point_indices`` is given, every field whose coord source
        matches ``target_coord_source()`` (i.e. the obs field's coord source)
        is sliced to those point indices before its stats are updated — this
        restricts the stats scope to e.g. train-only stations, closing a
        leakage bug where val/test stations contributed to the fit.

        Returns:
            ``(inner_by_field, output_transform)`` — ``inner_by_field`` maps
            each field name to its (un-nested) inner transform: the obs field
            maps to ``output_transform`` itself, every other field maps to its
            plain standardization transform. ``output_transform`` is the
            composed obs transform, usable standalone on obs-shaped tensors
            (e.g. model outputs).
        """
        obs_name = self._obs_field_name()
        obs_out = list(self.field_config.fields[obs_name].out)
        offset = 1e-3
        ttype = self.field_config.transform_type
        obs_pre = self._build_obs_pre(obs_out, ttype, offset)

        tcs = self.target_coord_source()
        tgt_idx = (
            None
            if target_point_indices is None
            else torch.as_tensor(list(target_point_indices), dtype=torch.long)
        )

        # Gather per-field stats over Field.values (obs stats after obs_pre).
        stats = {
            n: statistics.RunningFeatureStats(len(f.out), feature_dim=-1)
            for n, f in self.field_config.fields.items()
        }
        for idx in self._sample_indices(num_data_points, data_indices):
            fields = self[idx]["fields"]
            for n, f in fields.items():
                vals = f.values
                if tgt_idx is not None and self.field_config.fields[n].coords == tcs:
                    vals = vals[tgt_idx]
                vals = obs_pre(vals) if n == obs_name else vals
                stats[n].update(vals)

        output_transform = self._compose_output_transform(
            obs_pre, stats[obs_name], obs_out, ttype
        )
        inner: dict[str, T.Transform] = {}
        for n in self.field_config.fields:
            inner[n] = (
                output_transform if n == obs_name else stats[n].get_standard_transform()
            )
        return inner, output_transform

    def get_transform(
        self,
        num_data_points: int,
        data_indices: Sequence[int] | None = None,
        target_point_indices: Sequence[int] | None = None,
    ) -> tuple[T.NestedTransform, T.Transform]:
        """Compute data-driven per-field transforms (log/offset + standardize).

        Delegates to :meth:`compute_field_leaves` — see there for the stats
        gathering (and optional ``target_point_indices`` scoping) details.

        Returns:
            ``(post_transform, output_transform)`` — ``post_transform``
            standardizes every field's values (obs uses the composed
            pre-transform + standardization); ``output_transform`` is that
            same composed obs transform, usable standalone on obs-shaped
            tensors (e.g. model outputs).
        """
        inner, output_transform = self.compute_field_leaves(
            num_data_points, data_indices, target_point_indices
        )
        leaves = {n: T.FieldValueTransform(inner[n]) for n in inner}
        post_transform = T.NestedTransform({"fields": leaves})
        return post_transform, output_transform
