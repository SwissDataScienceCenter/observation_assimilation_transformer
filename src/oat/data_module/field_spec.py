"""Parsing of the config-driven field schema into typed specs (no validation here)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from oat.data_module.sources import SourceInfo

_COORD_AXES = ("x", "y", "t")


@dataclass(frozen=True)
class InputRef:
    kind: Literal["source", "coord"]
    source: str | None
    name: str


@dataclass(frozen=True)
class FieldSpec:
    name: str
    coords: str
    inputs: tuple[InputRef, ...]
    out: tuple[str, ...]
    query: bool


@dataclass(frozen=True)
class CoordSourceSpec:
    name: str
    kind: Literal["obs", "nwp", "regular"]
    res: float | None = None
    extent: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class FieldConfig:
    coord_sources: dict[str, CoordSourceSpec]
    fields: dict[str, FieldSpec]
    roles: dict[str, str]
    transform_type: str


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
    coord_sources: dict[str, CoordSourceSpec] = {}
    for name, spec in dict(cfg["coord_sources"]).items():
        spec = dict(spec)
        kind = spec["from"]
        if kind not in ("obs", "nwp", "regular"):
            msg = f"coord source {name!r} has unknown 'from': {kind!r}"
            raise ValueError(msg)
        extent = spec.get("extent")
        parsed_extent: tuple[float, float, float, float] | None = None
        if extent is not None:
            parsed_extent = (
                float(extent[0]),
                float(extent[1]),
                float(extent[2]),
                float(extent[3]),
            )
        coord_sources[name] = CoordSourceSpec(
            name=name,
            kind=kind,
            res=float(spec["res"]) if spec.get("res") is not None else None,
            extent=parsed_extent,
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
        fields[name] = FieldSpec(
            name=name,
            coords=spec["coords"],
            inputs=inputs,
            out=out,
            query=bool(spec.get("query", False)),
        )

    roles = {str(k): str(v) for k, v in dict(cfg.get("roles", {})).items()}
    return FieldConfig(
        coord_sources=coord_sources,
        fields=fields,
        roles=roles,
        transform_type=str(cfg["transform_type"]),
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
                if ref.name not in ("x", "y", "t"):
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
            r.source
            for r in field.inputs
            if r.kind == "source" and r.source in masked_sources
        }
        if used_masked:
            only_that_source = all(
                r.kind == "source" and r.source in used_masked for r in field.inputs
            )
            if len(used_masked) != 1 or not only_that_source:
                raise ValueError(
                    f"field {fname!r} may not mix masked-source inputs "
                    f"with other inputs"
                )

    for role, target in cfg.roles.items():
        if target not in cfg.fields:
            raise ValueError(f"role {role!r} names undeclared field {target!r}")

    target_field = cfg.roles.get("target")
    if target_field is not None:
        inputs = cfg.fields[target_field].inputs
        srcs = {r.source for r in inputs if r.kind == "source"}
        if not (
            len(srcs) == 1
            and next(iter(srcs)) in masked_sources
            and all(r.kind == "source" for r in inputs)
        ):
            raise ValueError(
                f"roles.target {target_field!r} must be the obs field "
                f"(masked source only)"
            )
