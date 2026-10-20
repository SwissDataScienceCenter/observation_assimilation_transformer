import pytest
import torch
from omegaconf import OmegaConf

from release.scrub_checkpoints import (
    ScrubError,
    check_no_leak,
    dataset_base,
    scrub_all,
    scrub_ckpt,
    scrub_config,
)

_BASE = OmegaConf.create(
    {
        "nwp_path": "${paths.data_root}/era5.zarr",
        "obs_path": "${paths.data_root}/obs.parquet",
    }
)


def _train_cfg():
    return OmegaConf.create(
        {
            "dataset": {
                "dataset_type": "era5_meteostat",
                "nwp_path": "/internal/x/era5.zarr",
                "obs_path": "/internal/x/obs.parquet",
                "ref_nwp_path": "/internal/x/nwp.zarr",
                "obs_vars": ["temp", "prcp"],
            },
            "logging": {"entity": "team-entity", "group": "oat_field"},
            "resume_from_checkpoint": "/internal/x/last.ckpt",
            "paths": {"run_root": "outputs/oat/2026"},
        }
    )


def test_scrub_config_rewrites_paths_and_logging():
    out = scrub_config(_train_cfg(), _BASE)
    raw = OmegaConf.to_container(out, resolve=False)
    assert raw["dataset"]["nwp_path"] == "${paths.data_root}/era5.zarr"
    assert raw["dataset"]["obs_path"] == "${paths.data_root}/obs.parquet"
    assert "ref_nwp_path" not in raw["dataset"]
    assert raw["logging"]["entity"] is None and raw["logging"]["group"] is None
    assert raw["resume_from_checkpoint"] is None
    assert raw["paths"] == {"data_root": "${oc.env:OAT_DATA_ROOT,data}"}


def test_scrubbed_config_resolves_with_env(monkeypatch):
    out = scrub_config(_train_cfg(), _BASE)
    monkeypatch.setenv("OAT_DATA_ROOT", "/d")
    assert out.dataset.nwp_path == "/d/era5.zarr"


def test_scrub_config_fails_on_left_absolute_path():
    cfg = _train_cfg()
    cfg.dataset.extra_path = "/internal/x/other.zarr"
    with pytest.raises(ScrubError, match="extra_path"):
        scrub_config(cfg, _BASE)


def test_scrub_config_fails_on_missing_ootg_keys():
    cfg = _train_cfg()
    cfg.split = {"train_station_indices": None}
    cfg.use_topo = True  # marks an OOTG config, see _is_ootg
    with pytest.raises(ScrubError, match="dataloader"):
        scrub_config(cfg, _BASE)


def test_scrub_ckpt_detaches_nodes_and_drops_callbacks():
    root = _train_cfg()
    names = root.dataset.obs_vars  # child node: pickles the full root
    ckpt = {
        "state_dict": {"w": torch.ones(2)},
        "callbacks": {"ModelCheckpoint": {"dirpath": "/internal/x"}},
        "hyper_parameters": {"output_var_names": names, "lr": 1e-3},
    }
    out = scrub_ckpt(ckpt)
    assert "callbacks" not in out
    assert list(out["hyper_parameters"]["output_var_names"]) == ["temp", "prcp"]
    assert out["hyper_parameters"]["lr"] == 1e-3
    assert torch.equal(out["state_dict"]["w"], torch.ones(2))
    import pickle

    assert b"/internal" not in pickle.dumps(out["hyper_parameters"])


@pytest.mark.parametrize(
    "text",
    [
        "p: /scr" "atch/x\n",
        "p: /us" "ers/x\n",
        "p: /cap" "stor/x\n",
        "p: /sto" "re_new/x\n",
        "a: me@meteo" "swiss.ch\n",
        "entity: sdsc" "-mch\n",
    ],
)
def test_check_no_leak(tmp_path, text):
    good = tmp_path / "a.yaml"
    good.write_text("x: 1\n")
    check_no_leak(good)
    bad = tmp_path / "b.yaml"
    bad.write_text(text)
    with pytest.raises(ScrubError, match=r"b\.yaml"):
        check_no_leak(bad)


def test_scrub_all_fails_on_missing_checkpoint(tmp_path):
    reg = tmp_path / "checkpoints.yaml"
    reg.write_text("era5_meteostat:\n  oat_temp_precip: fam/run/best.ckpt\n")
    with pytest.raises(ScrubError, match="not on disk"):
        scrub_all(reg, tmp_path / "src", tmp_path / "stage", tmp_path / "res")


def test_scrub_all_fails_on_absolute_registry_value(tmp_path):
    reg = tmp_path / "checkpoints.yaml"
    reg.write_text("era5_meteostat:\n  oat_temp_precip: /abs/run/best.ckpt\n")
    with pytest.raises(ScrubError, match="absolute"):
        scrub_all(reg, tmp_path / "src", tmp_path / "stage", tmp_path / "res")


def _raw(cfg):
    """Return the config as a dict with its interpolations left as text."""
    return OmegaConf.to_container(cfg, resolve=False)


def test_dataset_base_gives_full_domain_paths_to_swiss_ootg():
    base = _raw(dataset_base("cosmo_peakweather", ootg=True))
    assert "cosmo-1e_full_domain/nwp_extracted.zarr" in base["nwp_extracted_path"]
    assert "cosmo-1e_full_domain/" in base["nwp_path"]


def test_dataset_base_keeps_default_paths_for_other_runs():
    assert (
        "full_domain"
        not in _raw(dataset_base("cosmo_peakweather", ootg=False))["nwp_path"]
    )
    assert (
        "full_domain" not in _raw(dataset_base("era5_meteostat", ootg=True))["nwp_path"]
    )
