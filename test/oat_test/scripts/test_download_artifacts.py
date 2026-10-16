import hashlib
import io
import json
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from release.pack_artifacts import DATA_DIRS, DATA_FILES, pack_checkpoints, pack_data
from scripts import download_artifacts as dl


@pytest.fixture
def zenodo(monkeypatch):
    """A fake Zenodo API: records[rid] = {file name: bytes}."""
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    records: dict[str, dict[str, bytes]] = {}
    meta_override: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parts = self.path.strip("/").split("/")
            rid = parts[2] if len(parts) > 2 else ""
            if parts[:2] == ["api", "records"] and len(parts) == 3 and rid in records:
                meta = meta_override.get(rid) or {
                    "id": rid,
                    "files": [
                        {
                            "key": k,
                            "size": len(v),
                            "checksum": "md5:" + hashlib.md5(v).hexdigest(),
                        }
                        for k, v in records[rid].items()
                    ],
                }
                body = meta if isinstance(meta, bytes) else json.dumps(meta).encode()
            elif (
                len(parts) == 6
                and parts[3] == "files"
                and parts[5] == "content"
                and parts[4] in records.get(rid, {})
            ):
                body = records[rid][parts[4]]
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield SimpleNamespace(
        base=f"http://127.0.0.1:{srv.server_port}",
        records=records,
        meta_override=meta_override,
    )
    srv.shutdown()


def _record_from(folder):
    return {p.name: p.read_bytes() for p in folder.iterdir()}


def _packed(tmp_path):
    stage = tmp_path / "stage"
    run = stage / "checkpoints" / "oat_field" / "run_a"
    run.mkdir(parents=True)
    (run / "best.ckpt").write_bytes(b"ckpt")
    (stage / "results").mkdir()
    (stage / "results" / "t_metrics_test.csv").write_text("x\n")
    pack_checkpoints(stage, tmp_path / "up_c")
    root = tmp_path / "root"
    for name in DATA_DIRS:
        (root / name).mkdir(parents=True)
        (root / name / ".zgroup").write_text("{}")
    for name in DATA_FILES:
        (root / name).write_bytes(b"pq")
    pack_data(root, tmp_path / "up_d")
    return _record_from(tmp_path / "up_c"), _record_from(tmp_path / "up_d")


def test_round_trip_puts_files_at_the_default_roots(zenodo, tmp_path):
    ckpt_files, data_files = _packed(tmp_path)
    zenodo.records["11"] = ckpt_files
    zenodo.records["22"] = data_files
    dest = tmp_path / "dest"
    assert dl.download_record(zenodo.base, "11", dest) == 2
    assert dl.download_record(zenodo.base, "22", dest) == 4
    assert (dest / "checkpoints/oat_field/run_a/best.ckpt").read_bytes() == b"ckpt"
    assert (dest / "data/results/t_metrics_test.csv").is_file()
    assert (dest / "data" / DATA_DIRS[0] / ".zgroup").is_file()
    assert (dest / "data" / DATA_FILES[0]).read_bytes() == b"pq"
    assert not (dest / ".oat_download").exists()
    assert not list(dest.rglob("*.tar"))


def test_md5_mismatch_fails_and_leaves_no_part_file(zenodo, tmp_path):
    ckpt_files, _ = _packed(tmp_path)
    zenodo.records["11"] = ckpt_files
    zenodo.meta_override["11"] = {
        "files": [{"key": k, "checksum": "md5:" + "0" * 32} for k in ckpt_files]
    }
    dest = tmp_path / "dest"
    with pytest.raises(dl.DownloadError, match="MD5"):
        dl.download_record(zenodo.base, "11", dest)
    assert not list(dest.rglob("*.part"))
    assert not (dest / "checkpoints").exists()


def test_sha256_mismatch_after_unpack_fails(zenodo, tmp_path):
    ckpt_files, _ = _packed(tmp_path)
    digest, _ = ckpt_files["SHA256SUMS"].decode().splitlines()[0].split(maxsplit=1)
    bad = ckpt_files["SHA256SUMS"].decode().replace(digest, "f" * 64)
    zenodo.records["11"] = {**ckpt_files, "SHA256SUMS": bad.encode()}
    with pytest.raises(dl.DownloadError, match="SHA256"):
        dl.download_record(zenodo.base, "11", tmp_path / "dest")


def test_second_run_after_a_failed_run_works(zenodo, tmp_path):
    ckpt_files, _ = _packed(tmp_path)
    zenodo.records["11"] = {**ckpt_files, "SHA256SUMS": b"0" * 64 + b"  x\n"}
    dest = tmp_path / "dest"
    with pytest.raises(dl.DownloadError):
        dl.download_record(zenodo.base, "11", dest)
    zenodo.records["11"] = ckpt_files
    assert dl.download_record(zenodo.base, "11", dest) == 2


def test_unsafe_tar_member_is_refused(zenodo, tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo("../evil.txt")
        info.size = 4
        tar.addfile(info, io.BytesIO(b"evil"))
    zenodo.records["11"] = {"bad.tar": buf.getvalue(), "SHA256SUMS": b""}
    dest = tmp_path / "dest"
    with pytest.raises(dl.DownloadError, match=r"bad\.tar"):
        dl.download_record(zenodo.base, "11", dest)
    assert not (tmp_path / "evil.txt").exists()


def test_unknown_metadata_shape_gives_clear_error(zenodo, tmp_path):
    zenodo.records["11"] = {"a": b"x"}
    zenodo.meta_override["11"] = {"entries": []}
    with pytest.raises(dl.DownloadError, match="no file list"):
        dl.download_record(zenodo.base, "11", tmp_path / "d")
    zenodo.meta_override["11"] = {"files": [{"key": "a", "checksum": "sha1:abc"}]}
    with pytest.raises(dl.DownloadError, match="MD5"):
        dl.download_record(zenodo.base, "11", tmp_path / "d")


def test_record_without_sums_file_is_refused(zenodo, tmp_path):
    zenodo.records["11"] = {"a.parquet": b"x"}
    with pytest.raises(dl.DownloadError, match="SHA256SUMS"):
        dl.download_record(zenodo.base, "11", tmp_path / "d")


def test_http_error_names_the_url(zenodo, tmp_path):
    with pytest.raises(dl.DownloadError, match="/api/records/404"):
        dl.download_record(zenodo.base, "404", tmp_path / "d")


def test_main_without_record_id_stops_with_message(capsys, tmp_path):
    assert dl.main(["--what", "checkpoints", "--dest", str(tmp_path)]) == 1
    assert "--checkpoints-record" in capsys.readouterr().err


def test_main_uses_record_override_and_sandbox(zenodo, tmp_path, monkeypatch):
    ckpt_files, _ = _packed(tmp_path)
    zenodo.records["11"] = ckpt_files
    monkeypatch.setitem(dl.SERVERS, "sandbox", zenodo.base)
    argv = ["--what", "checkpoints", "--sandbox", "--checkpoints-record", "11"]
    assert dl.main([*argv, "--dest", str(tmp_path / "dest")]) == 0
    assert (tmp_path / "dest/checkpoints/oat_field/run_a/best.ckpt").is_file()


def _tar_with(info, data=b""):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_non_json_metadata_gives_clear_error(zenodo, tmp_path):
    zenodo.records["11"] = {"a": b"x"}
    zenodo.meta_override["11"] = b"<html>proxy</html>"
    with pytest.raises(dl.DownloadError, match="not JSON"):
        dl.download_record(zenodo.base, "11", tmp_path / "d")


def test_symlink_tar_member_is_refused(zenodo, tmp_path):
    info = tarfile.TarInfo("data/link")
    info.type = tarfile.SYMTYPE
    info.linkname = "/etc/passwd"
    zenodo.records["11"] = {"bad.tar": _tar_with(info), "SHA256SUMS": b""}
    with pytest.raises(dl.DownloadError, match=r"bad\.tar"):
        dl.download_record(zenodo.base, "11", tmp_path / "d")
    assert not (tmp_path / "d/data/link").exists()


def test_absolute_tar_member_is_refused(zenodo, tmp_path):
    info = tarfile.TarInfo("/abs.txt")
    info.size = 1
    zenodo.records["11"] = {"bad.tar": _tar_with(info, b"x"), "SHA256SUMS": b""}
    with pytest.raises(dl.DownloadError, match=r"bad\.tar"):
        dl.download_record(zenodo.base, "11", tmp_path / "d")
    assert not (tmp_path / "d/abs.txt").exists()


def test_empty_sums_file_is_refused(zenodo, tmp_path):
    ckpt_files, _ = _packed(tmp_path)
    zenodo.records["11"] = {**ckpt_files, "SHA256SUMS": b""}
    with pytest.raises(dl.DownloadError, match="lists no files"):
        dl.download_record(zenodo.base, "11", tmp_path / "d")
