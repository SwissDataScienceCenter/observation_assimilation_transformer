import tarfile

import pytest

from release.pack_artifacts import DATA_DIRS, DATA_FILES, pack_checkpoints, pack_data
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
