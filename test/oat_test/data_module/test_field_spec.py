import pytest

from oat.data_module.field_spec import (
    InputRef,
    parse_field_config,
    parse_input_ref,
)


def test_parse_input_ref_source_and_coord():
    assert parse_input_ref("nwp.T_2M") == InputRef(
        kind="source", source="nwp", name="T_2M"
    )
    assert parse_input_ref("coords.x") == InputRef(kind="coord", source=None, name="x")


def test_parse_input_ref_rejects_bad_axis_and_shape():
    with pytest.raises(ValueError):
        parse_input_ref("coords.q")  # not x/y/t
    with pytest.raises(ValueError):
        parse_input_ref("nwp")  # missing dotted var
    with pytest.raises(ValueError):
        parse_input_ref("a.b.c")  # too many parts


def test_parse_field_config_full():
    cfg = {
        "coord_sources": {
            "stations": {"from": "obs"},
            "nwp_grid": {"from": "nwp"},
        },
        "fields": {
            "obs": {"coords": "stations", "inputs": ["obs.temperature"]},
            "nwp": {
                "coords": "stations",
                "inputs": ["nwp.T_2M", "nwp.wind_speed"],
                "query": True,
            },
            "geo": {
                "coords": "stations",
                "inputs": ["coords.x", "coords.y", "topo.DEM"],
                "out": ["easting", "northing", "dem"],
                "query": True,
            },
        },
        "roles": {"target": "obs"},
        "transform_type": "real",
    }
    fc = parse_field_config(cfg)
    assert fc.transform_type == "real"
    assert fc.coord_sources == frozenset({"stations", "nwp_grid"})
    assert fc.nwp_coord_source == "nwp_grid"
    # out defaults to input leaf names when omitted
    assert fc.fields["nwp"].out == ("T_2M", "wind_speed")
    assert fc.fields["nwp"].query is True
    assert fc.fields["obs"].query is False  # default
    assert fc.fields["geo"].out == ("easting", "northing", "dem")
    assert fc.fields["geo"].inputs[2] == InputRef(
        kind="source", source="topo", name="DEM"
    )
    # k defaults to 1 (nearest) when omitted
    assert fc.fields["obs"].k == 1
    assert fc.fields["nwp"].k == 1


def test_parse_field_config_k_override():
    cfg = {
        "coord_sources": {"stations": {"from": "obs"}, "nwp_grid": {"from": "nwp"}},
        "fields": {
            "nwp": {
                "coords": "stations",
                "inputs": ["nwp.T_2M"],
                "query": True,
                "k": 4,
            },
        },
        "roles": {"target": "nwp"},
        "transform_type": "real",
    }
    fc = parse_field_config(cfg)
    assert fc.fields["nwp"].k == 4
