"""Parsing of the config-driven field schema into typed specs (no validation here)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

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
        raise ValueError(f"input ref must be '<source>.<var>' or 'coords.<axis>', got {ref!r}")
    head, tail = parts
    if head == "coords":
        if tail not in _COORD_AXES:
            raise ValueError(f"coords axis must be one of {_COORD_AXES}, got {tail!r}")
        return InputRef(kind="coord", source=None, name=tail)
    return InputRef(kind="source", source=head, name=tail)


def parse_field_config(cfg: Mapping[str, Any]) -> FieldConfig:
    coord_sources: dict[str, CoordSourceSpec] = {}
    for name, spec in dict(cfg["coord_sources"]).items():
        spec = dict(spec)
        kind = spec["from"]
        if kind not in ("obs", "nwp", "regular"):
            raise ValueError(f"coord source {name!r} has unknown 'from': {kind!r}")
        extent = spec.get("extent")
        coord_sources[name] = CoordSourceSpec(
            name=name,
            kind=kind,
            res=float(spec["res"]) if spec.get("res") is not None else None,
            extent=tuple(float(v) for v in extent) if extent is not None else None,
        )

    fields: dict[str, FieldSpec] = {}
    for name, spec in dict(cfg["fields"]).items():
        spec = dict(spec)
        inputs = tuple(parse_input_ref(r) for r in spec["inputs"])
        out = spec.get("out")
        out = tuple(str(o) for o in out) if out is not None else tuple(r.name for r in inputs)
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
