"""Parsing of the config-driven field schema into typed specs (no validation here)."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

INTERP_MODES = ("nearest", "linear")

from oat.data_module.sources import SourceInfo

_COORD_AXES = ("x", "y", "t")


@dataclass(frozen=True)
class InputRef:
    kind: Literal["source", "coord"]
    source: str | None
    name: str


class SourceRef(NamedTuple):
    """An ``InputRef`` narrowed to the source-kind case (source/name non-None)."""

    source: str
    name: str


@dataclass(frozen=True)
class FieldSpec:
    name: str
    coords: str
    inputs: tuple[InputRef, ...]
    out: tuple[str, ...]
    query: bool
    interp: str = "nearest"

    @property
    def source_refs(self) -> tuple[SourceRef, ...]:
        """The subset of ``inputs`` that read from a source (not a coord)."""
        return tuple(
            SourceRef(r.source, r.name)
            for r in self.inputs
            if r.kind == "source" and r.source is not None
        )


@dataclass(frozen=True)
class GriddedSpec:
    """Regular 2D grid derived from the bounding box of an irregular coord source."""

    res: float
    lon_min: float | None = None
    lon_max: float | None = None
    lat_min: float | None = None
    lat_max: float | None = None


@dataclass(frozen=True)
class FieldConfig:
    coord_sources: frozenset[str]
    nwp_coord_source: str | None
    fields: dict[str, FieldSpec]
    roles: dict[str, str]
    gridded_sources: dict[str, GriddedSpec] = dataclasses.field(default_factory=dict)


def parse_input_ref(ref: str) -> InputRef:
    parts = str(ref).split(".")
    if len(parts) != 2:
        msg = f"input ref must be '<source>.<var>' or 'coords.<axis>', got {ref!r}"
        raise ValueError(msg)
    head, tail = parts
    if head == "coords":
        if tail not in _COORD_AXES:
            msg = f"coords axis must be one of {_COORD_AXES}, got {tail!r}"
            raise ValueError(msg)
        return InputRef(kind="coord", source=None, name=tail)
    return InputRef(kind="source", source=head, name=tail)


def parse_field_config(cfg: Mapping[str, Any]) -> FieldConfig:
    coord_sources: set[str] = set()
    nwp_coord_source: str | None = None
    gridded_sources: dict[str, GriddedSpec] = {}
    for name, spec in dict(cfg["coord_sources"]).items():
        spec = dict(spec)
        kind = spec["from"]
        if kind not in ("obs", "nwp"):
            msg = f"coord source {name!r} has unknown 'from': {kind!r}"
            raise ValueError(msg)
        coord_sources.add(name)
        if kind == "nwp" and nwp_coord_source is None:
            nwp_coord_source = name
        gridded_cfg = spec.get("gridded")
        if gridded_cfg is not None:
            g = dict(gridded_cfg)
            bounds = g.get("bounds")
            if bounds is not None:
                bounds = dict(bounds)
            gridded_sources[name] = GriddedSpec(
                res=float(g["res"]),
                lon_min=float(bounds["lon_min"]) if bounds else None,
                lon_max=float(bounds["lon_max"]) if bounds else None,
                lat_min=float(bounds["lat_min"]) if bounds else None,
                lat_max=float(bounds["lat_max"]) if bounds else None,
            )

    fields: dict[str, FieldSpec] = {}
    for name, spec in dict(cfg["fields"]).items():
        spec = dict(spec)
        inputs = tuple(parse_input_ref(r) for r in spec["inputs"])
        out_spec = spec.get("out")
        if out_spec is not None:
            out = tuple(str(o) for o in out_spec)
        else:
            out = tuple(r.name for r in inputs)
        interp = str(spec.get("interp", "nearest"))
        if interp not in INTERP_MODES:
            raise ValueError(
                f"field {name!r}: interp must be one of {INTERP_MODES}, "
                f"got {interp!r}"
            )
        fields[name] = FieldSpec(
            name=name,
            coords=spec["coords"],
            inputs=inputs,
            out=out,
            query=bool(spec.get("query", False)),
            interp=interp,
        )

    roles = {str(k): str(v) for k, v in dict(cfg.get("roles", {})).items()}
    return FieldConfig(
        coord_sources=frozenset(coord_sources),
        nwp_coord_source=nwp_coord_source,
        fields=fields,
        roles=roles,
        gridded_sources=gridded_sources,
    )


def validate_field_config(cfg: FieldConfig, sources: dict[str, SourceInfo]) -> None:
    masked_sources = {name for name, s in sources.items() if s.has_mask}

    for fname, field in cfg.fields.items():
        if not fname:
            raise ValueError("field names must be non-empty")
        if field.coords not in cfg.coord_sources:
            raise ValueError(
                f"field {fname!r} uses unknown coord source {field.coords!r}"
            )
        for ref in field.inputs:
            if ref.kind == "coord":
                if ref.name not in _COORD_AXES:
                    raise ValueError(
                        f"field {fname!r} has bad coords axis {ref.name!r}"
                    )
            else:
                if ref.source not in sources:
                    raise ValueError(
                        f"field {fname!r} refers to unknown source {ref.source!r}"
                    )
                if ref.name not in sources[ref.source].available_vars:
                    raise ValueError(
                        f"field {fname!r}: var {ref.name!r} "
                        f"not in source {ref.source!r}"
                    )
        if len(field.out) != len(field.inputs):
            raise ValueError(
                f"field {fname!r}: out has {len(field.out)} names "
                f"for {len(field.inputs)} inputs"
            )
        if len(set(field.out)) != len(field.out):
            raise ValueError(
                f"field {fname!r}: output channel names must be unique: {field.out}"
            )

        used_masked = {
            r.source for r in field.source_refs if r.source in masked_sources
        }
        if used_masked:
            only_that_source = len(field.source_refs) == len(field.inputs) and all(
                r.source in used_masked for r in field.source_refs
            )
            if len(used_masked) != 1 or not only_that_source:
                raise ValueError(
                    f"field {fname!r} may not mix masked-source inputs "
                    f"with other inputs"
                )
            if field.interp != "nearest":
                raise ValueError(
                    f"field {fname!r} has interp={field.interp!r} but "
                    f"references masked source(s) {sorted(used_masked)}: "
                    f"interpolating a mask is ill-defined"
                )

    for role, target in cfg.roles.items():
        if target not in cfg.fields:
            raise ValueError(f"role {role!r} names undeclared field {target!r}")

    target_field = cfg.roles.get("target")
    if target_field is not None:
        target_spec = cfg.fields[target_field]
        srcs = {r.source for r in target_spec.source_refs}
        if not (
            len(srcs) == 1
            and next(iter(srcs)) in masked_sources
            and len(target_spec.source_refs) == len(target_spec.inputs)
        ):
            raise ValueError(
                f"roles.target {target_field!r} must be the obs field "
                f"(masked source only)"
            )
