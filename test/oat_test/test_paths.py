from pathlib import Path

import pytest

from oat import paths
from oat.predictor_module.checkpoint_utils import resolve_checkpoint


def test_data_root_uses_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OAT_DATA_ROOT", str(tmp_path))
    assert paths.data_root() == tmp_path


def test_data_root_default_is_repo_data(monkeypatch):
    monkeypatch.delenv("OAT_DATA_ROOT", raising=False)
    assert paths.data_root() == paths.REPO_ROOT / "data"


def test_checkpoint_root_default_is_repo_checkpoints(monkeypatch):
    monkeypatch.delenv("OAT_CKPT_ROOT", raising=False)
    assert paths.checkpoint_root() == paths.REPO_ROOT / "checkpoints"


def test_repo_root_holds_configs():
    assert (paths.REPO_ROOT / "configs" / "config.yaml").is_file()
    assert paths.REGISTRY_PATH.is_file()


@pytest.fixture
def registry(tmp_path):
    p = tmp_path / "checkpoints.yaml"
    p.write_text(
        "ds:\n"
        "  interpolator: interpolator\n"
        "  rel: fam/run/best.ckpt\n"
        "  abs: /old/fam/run/best.ckpt\n"
    )
    return p


def test_resolve_checkpoint_joins_relative_value(registry, monkeypatch, tmp_path):
    monkeypatch.setenv("OAT_CKPT_ROOT", str(tmp_path / "ck"))
    got = resolve_checkpoint("ds", "rel", registry_path=registry)
    assert got == str(tmp_path / "ck" / "fam/run/best.ckpt")


def test_resolve_checkpoint_keeps_absolute_value(registry):
    assert resolve_checkpoint("ds", "abs", registry_path=registry) == (
        "/old/fam/run/best.ckpt"
    )


def test_resolve_checkpoint_keeps_sentinel(registry):
    assert resolve_checkpoint("ds", "interpolator", registry_path=registry) == (
        "interpolator"
    )


def test_tracked_registry_is_relative():
    from omegaconf import OmegaConf

    reg = OmegaConf.to_container(OmegaConf.load(paths.REGISTRY_PATH))
    for models in reg.values():
        for key, value in (models or {}).items():
            if value is None or value == key:
                continue
            assert not Path(value).is_absolute(), (key, value)
