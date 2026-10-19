import shutil
import tarfile

import pytest

from release.pack_artifacts import (
    DATA_DIRS,
    DATA_FILES,
    SWISS_README,
    SWISS_STORES,
    pack_checkpoints,
    pack_data,
    pack_swiss,
)
from scripts.download_artifacts import SUMS_NAME, file_digest


def _sums(path):
    out = {}
    for line in path.read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        out[rel] = digest
    return out


def _stage(tmp_path):
    stage = tmp_path / "stage"
    run = stage / "checkpoints" / "oat_field" / "run_a"
    run.mkdir(parents=True)
    (run / "best.ckpt").write_bytes(b"ckpt")
    (run / "config.yaml").write_text("a: 1\n")
    (stage / "results").mkdir()
    (stage / "results" / "tag_metrics_test.csv").write_text("x\n")
    return stage


def test_pack_checkpoints_member_names_and_sums(tmp_path):
    out = tmp_path / "up"
    written = pack_checkpoints(_stage(tmp_path), out)
    assert [p.name for p in written] == ["checkpoints.tar", "results.tar", SUMS_NAME]
    with tarfile.open(out / "checkpoints.tar") as tar:
        assert sorted(tar.getnames()) == [
            "checkpoints/oat_field/run_a/best.ckpt",
            "checkpoints/oat_field/run_a/config.yaml",
        ]
    with tarfile.open(out / "results.tar") as tar:
        assert tar.getnames() == ["data/results/tag_metrics_test.csv"]
    for name in ("checkpoints.tar", "results.tar"):
        with tarfile.open(out / name) as tar:
            assert all(m.uname == "" and m.uid == 0 for m in tar.getmembers())
    sums = _sums(out / SUMS_NAME)
    assert set(sums) == {
        "checkpoints/oat_field/run_a/best.ckpt",
        "checkpoints/oat_field/run_a/config.yaml",
        "data/results/tag_metrics_test.csv",
    }
    src = tmp_path / "stage" / "checkpoints" / "oat_field" / "run_a" / "best.ckpt"
    assert sums["checkpoints/oat_field/run_a/best.ckpt"] == file_digest(src, "sha256")


def _data_root(tmp_path):
    real = tmp_path / "real"
    root = tmp_path / "root"
    root.mkdir()
    for name in DATA_DIRS:
        (real / name / "data").mkdir(parents=True)
        (real / name / ".zgroup").write_text("{}")
        (real / name / "data" / "0.0").write_bytes(b"\x00\x01")
        (root / name).symlink_to(real / name)  # the team root holds links
    for name in DATA_FILES:
        (real / name).write_bytes(b"parquet")
        (root / name).symlink_to(real / name)
    return root


def test_pack_data_follows_links_and_copies_files(tmp_path):
    out = tmp_path / "up"
    written = pack_data(_data_root(tmp_path), out)
    assert sorted(p.name for p in written) == sorted(
        [f"{n}.tar" for n in DATA_DIRS] + list(DATA_FILES) + [SUMS_NAME]
    )
    with tarfile.open(out / f"{DATA_DIRS[0]}.tar") as tar:
        members = {m.name: m for m in tar.getmembers()}
    assert set(members) == {
        f"data/{DATA_DIRS[0]}/.zgroup",
        f"data/{DATA_DIRS[0]}/data/0.0",
    }
    assert all(m.isfile() for m in members.values())  # no link members
    for name in DATA_FILES:
        assert (out / name).is_file() and not (out / name).is_symlink()
    sums = _sums(out / SUMS_NAME)
    assert f"data/{DATA_FILES[0]}" in sums
    assert f"data/{DATA_DIRS[1]}/data/0.0" in sums


def test_pack_data_fails_on_missing_input(tmp_path):
    root = _data_root(tmp_path)
    (root / DATA_FILES[1]).unlink()
    with pytest.raises(FileNotFoundError, match=DATA_FILES[1]):
        pack_data(root, tmp_path / "up")


def test_pack_checkpoints_fails_on_empty_stage(tmp_path):
    (tmp_path / "stage" / "checkpoints").mkdir(parents=True)
    (tmp_path / "stage" / "results").mkdir()
    with pytest.raises(FileNotFoundError, match="no files"):
        pack_checkpoints(tmp_path / "stage", tmp_path / "up")


def _fake_swiss_root(tmp_path):
    root = tmp_path / "swiss"
    for store in SWISS_STORES:
        if store == "peakweather":
            for name in ("disclaimer.txt", "a.csv", "b.csv"):
                (root / store).mkdir(parents=True, exist_ok=True)
                (root / store / name).write_bytes(b"p" * 40)
        else:
            (root / store / "0").mkdir(parents=True)
            (root / store / ".zgroup").write_bytes(b"{}")
            (root / store / "0" / "0.0").write_bytes(b"z" * 10)
    return root


def test_pack_swiss_splits_a_large_store_into_complete_tars(tmp_path):
    root = _fake_swiss_root(tmp_path)
    written = pack_swiss(root, tmp_path / "up", max_bytes=16000)
    pw = sorted(p.name for p in written if p.name.startswith("peakweather"))
    assert pw == ["peakweather.000.tar", "peakweather.001.tar"]
    names = [
        n for p in written if p.suffix == ".tar" for n in tarfile.open(p).getnames()
    ]
    assert len(names) == len(set(names))
    assert set(_sums(tmp_path / "up" / SUMS_NAME)) == set(names)
    assert all(n.startswith("data/") for n in names)
    assert (tmp_path / "up/README.md").read_text() == SWISS_README


def test_pack_swiss_needs_every_store(tmp_path):
    root = _fake_swiss_root(tmp_path)
    shutil.rmtree(root / "peakweather")
    with pytest.raises(FileNotFoundError):
        pack_swiss(root, tmp_path / "up")


def test_pack_swiss_parts_stay_within_max_bytes(tmp_path):
    root = _fake_swiss_root(tmp_path)
    for i in range(12):
        (root / "peakweather" / f"big{i}.bin").write_bytes(b"q" * 3000)
    written = pack_swiss(root, tmp_path / "up", max_bytes=30000)
    parts = [p for p in written if p.name.startswith("peakweather")]
    assert len(parts) > 1
    for p in parts:
        assert p.stat().st_size <= 30000


def test_pack_swiss_refuses_an_out_folder_with_old_tars(tmp_path):
    root = _fake_swiss_root(tmp_path)
    (tmp_path / "up").mkdir()
    (tmp_path / "up" / "old.tar").write_bytes(b"")
    with pytest.raises(FileExistsError, match="tar files"):
        pack_swiss(root, tmp_path / "up")
    assert sorted(p.name for p in (tmp_path / "up").iterdir()) == ["old.tar"]


def test_pack_swiss_refuses_a_file_larger_than_the_part_size(tmp_path):
    root = _fake_swiss_root(tmp_path)
    (root / "peakweather" / "huge.bin").write_bytes(b"h" * 20000)
    with pytest.raises(ValueError, match="larger than the part size"):
        pack_swiss(root, tmp_path / "up", max_bytes=16000)


def test_pack_swiss_refuses_an_internal_string_in_zarr_metadata(tmp_path):
    root = _fake_swiss_root(tmp_path)
    bad = root / "cosmo-1e_rechunked/topo.zarr/.zattrs"
    bad.write_text('{"source": "/' + 'users/someone/topo.zarr"}')
    with pytest.raises(ValueError, match=r"topo\.zarr/\.zattrs"):
        pack_swiss(root, tmp_path / "up")
    assert not (tmp_path / "up").exists()
