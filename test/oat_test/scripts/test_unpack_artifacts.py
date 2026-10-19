import io
import tarfile

import pytest

from release.pack_artifacts import SWISS_STORES, pack_swiss
from scripts.download_artifacts import DownloadError
from scripts.unpack_artifacts import main


@pytest.fixture
def packed(tmp_path):
    root = tmp_path / "swiss"
    for store in SWISS_STORES:
        (root / store).mkdir(parents=True)
        (root / store / "f.bin").write_bytes(store.encode())
    (root / "peakweather" / "disclaimer.txt").write_text("PeakWeather")
    for i in range(3):
        (root / "peakweather" / f"extra{i}.bin").write_bytes(b"e" * 40)
    pack_swiss(root, tmp_path / "up", max_bytes=16000)
    assert len(list((tmp_path / "up").glob("peakweather.*.tar"))) > 1
    return tmp_path / "up"


def _rewrite_one_member(packed):
    path = packed / "peakweather.000.tar"
    with tarfile.open(path) as tar:
        items = [(m, tar.extractfile(m).read()) for m in tar.getmembers()]
    with tarfile.open(path, "w") as tar:
        for member, data in items:
            data = b"x" * len(data)
            tar.addfile(member, io.BytesIO(data))


def test_unpack_restores_every_file_and_keeps_the_tars(packed, tmp_path):
    assert main(["--src", str(packed), "--dest", str(tmp_path / "d")]) == 0
    assert (tmp_path / "d/data/peakweather/disclaimer.txt").is_file()
    assert list(packed.glob("*.tar"))
    for line in (packed / "SHA256SUMS").read_text().splitlines():
        assert (tmp_path / "d" / line.split(maxsplit=1)[1]).is_file()


def test_unpack_stops_before_writing_when_a_target_exists(packed, tmp_path):
    (tmp_path / "d/data/peakweather").mkdir(parents=True)
    with pytest.raises(DownloadError, match="exists"):
        main(["--src", str(packed), "--dest", str(tmp_path / "d")])
    assert not (tmp_path / "d/data/cosmo-1e_rechunked").exists()


def test_unpack_fails_on_a_changed_file(packed, tmp_path):
    _rewrite_one_member(packed)
    with pytest.raises(DownloadError, match="SHA256"):
        main(["--src", str(packed), "--dest", str(tmp_path / "d")])
