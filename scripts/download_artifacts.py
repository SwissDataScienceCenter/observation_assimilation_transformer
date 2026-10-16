"""Download the released checkpoints, result files and Europe dataset.

The files come from two Zenodo records. For each file the script checks the
MD5 sum that Zenodo gives. Then it unpacks the tar files below --dest and
checks every file that the SHA256SUMS file of the record lists. It
fails on a checksum mismatch or an HTTP error. It uses only the standard
library.

The files go to the folders that the path defaults read:
    <dest>/checkpoints/...   the checkpoint root (OAT_CKPT_ROOT)
    <dest>/data/...          the data root (OAT_DATA_ROOT)
    <dest>/data/results/...  the result files of the paper table

Usage (from the repository root):
    python -m scripts.download_artifacts --what all --dest .
    python -m scripts.download_artifacts --what checkpoints --sandbox \\
        --checkpoints-record <id>
"""

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from http.client import HTTPException, HTTPResponse
from pathlib import Path
from typing import cast

SERVERS = {"zenodo": "https://zenodo.org", "sandbox": "https://sandbox.zenodo.org"}
# The record IDs are set when the records are reserved (release step 2).
RECORD_IDS: dict[str, dict[str, str | None]] = {
    "zenodo": {"checkpoints": None, "data": None},
    "sandbox": {"checkpoints": None, "data": None},
}
SUMS_NAME = "SHA256SUMS"
_CHUNK = 1 << 20
_TIMEOUT_S = 60
_WORK = ".oat_download"


class DownloadError(RuntimeError):
    """A file cannot be downloaded, or it does not match its checksum."""


def file_digest(path: Path, algorithm: str) -> str:
    """Return the hex digest of the file at *path* (``md5`` or ``sha256``)."""
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _open(url: str) -> HTTPResponse:
    req = urllib.request.Request(url, headers={"User-Agent": "oat-download"})
    try:
        return cast(HTTPResponse, urllib.request.urlopen(req, timeout=_TIMEOUT_S))
    except (OSError, HTTPException) as exc:  # URLError, HTTPError: OSError
        raise DownloadError(f"{url}: {exc}") from exc


def record_files(base: str, record_id: str) -> list[tuple[str, str]]:
    """Return (file name, MD5 hex) for each file of a published record."""
    url = f"{base}/api/records/{record_id}"
    with _open(url) as resp:
        try:
            meta = json.load(resp)
        except (ValueError, OSError, HTTPException) as exc:
            raise DownloadError(f"{url}: metadata is not JSON ({exc})") from exc
    files = meta.get("files") if isinstance(meta, dict) else None
    if not isinstance(files, list) or not files:
        raise DownloadError(f"{url}: no file list in the record metadata")
    out: list[tuple[str, str]] = []
    for entry in files:
        key = entry.get("key") if isinstance(entry, dict) else None
        checksum = str(entry.get("checksum", "")) if isinstance(entry, dict) else ""
        if not isinstance(key, str) or not key or "/" in key or key in {".", ".."}:
            raise DownloadError(f"{url}: file entry without a safe name: {entry}")
        if not checksum.startswith("md5:"):
            raise DownloadError(f"{url}: {key} has no MD5 checksum")
        out.append((key, checksum.removeprefix("md5:")))
    return out


def fetch(url: str, path: Path, md5: str) -> None:
    """Download *url* to *path*. Fail when the MD5 sum differs."""
    part = path.with_name(path.name + ".part")
    try:
        with _open(url) as resp, part.open("wb") as f:
            shutil.copyfileobj(resp, f, _CHUNK)
    except (OSError, HTTPException) as exc:
        part.unlink(missing_ok=True)
        raise DownloadError(f"{url}: {exc}") from exc
    got = file_digest(part, "md5")
    if got != md5:
        part.unlink()
        raise DownloadError(f"{path.name}: MD5 {got} is not the expected {md5}")
    part.replace(path)


def _check_member(tar_name: str, member: tarfile.TarInfo) -> None:
    parts = Path(member.name).parts
    if (
        member.issym()
        or member.islnk()
        or ".." in parts
        or not parts
        or parts[0] not in {"checkpoints", "data"}
        or member.name.startswith("/")
    ):
        raise DownloadError(f"{tar_name}: unsafe member {member.name!r}")


def unpack(path: Path, dest: Path) -> None:
    """Extract a ``.tar`` file into *dest*, or move another file to dest/data/."""
    if path.name.endswith(".tar"):
        try:
            with tarfile.open(path) as tar:
                for member in tar.getmembers():
                    _check_member(path.name, member)
                tar.extractall(dest, filter="data")  # refuses unsafe members
        except tarfile.TarError as exc:
            raise DownloadError(f"{path.name}: {exc}") from exc
        path.unlink()
    else:
        (dest / "data").mkdir(parents=True, exist_ok=True)
        path.replace(dest / "data" / path.name)


def verify_sums(sums: Path, dest: Path) -> int:
    """Check each ``<sha256>  <path>`` line against dest/<path>. Return the count."""
    count = 0
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise DownloadError(f"{SUMS_NAME}: bad line {line!r}")
        digest, rel = fields
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise DownloadError(f"{SUMS_NAME}: unsafe path {rel!r}")
        target = dest / rel
        if not target.is_file():
            raise DownloadError(f"{rel}: in {SUMS_NAME}, but not on disk")
        if file_digest(target, "sha256") != digest:
            raise DownloadError(f"{rel}: SHA256 does not match {SUMS_NAME}")
        count += 1
    if count == 0:
        raise DownloadError(f"{SUMS_NAME} lists no files")
    return count


def download_record(base: str, record_id: str, dest: Path) -> int:
    """Download, unpack and check one record. Return the number of checked files."""
    files = record_files(base, record_id)
    names = [key for key, _ in files]
    if SUMS_NAME not in names:
        raise DownloadError(f"record {record_id}: no {SUMS_NAME} file")
    work = dest / _WORK / record_id
    if work.exists():
        shutil.rmtree(work)  # left by a failed run
    work.mkdir(parents=True)
    for key, md5 in files:
        print(f"download {key}", file=sys.stderr)
        fetch(f"{base}/api/records/{record_id}/files/{key}/content", work / key, md5)
    for key in names:
        if key != SUMS_NAME:
            unpack(work / key, dest)
    count = verify_sums(work / SUMS_NAME, dest)
    shutil.rmtree(work)
    if not any((dest / _WORK).iterdir()):
        (dest / _WORK).rmdir()
    return count


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download the OAT release from Zenodo.")
    p.add_argument("--what", choices=["checkpoints", "data", "all"], required=True)
    p.add_argument("--dest", type=Path, default=Path("."))
    p.add_argument("--sandbox", action="store_true", help="Use sandbox.zenodo.org.")
    p.add_argument("--checkpoints-record", default=None, help="Record ID override.")
    p.add_argument("--data-record", default=None, help="Record ID override.")
    args = p.parse_args(argv)
    server = "sandbox" if args.sandbox else "zenodo"
    override = {"checkpoints": args.checkpoints_record, "data": args.data_record}
    whats = ["checkpoints", "data"] if args.what == "all" else [args.what]
    try:
        for what in whats:
            record_id = override[what] or RECORD_IDS[server][what]
            if record_id is None:
                raise DownloadError(
                    f"no {server} record ID for {what!r} yet; "
                    f"give it with --{what}-record"
                )
            count = download_record(SERVERS[server], record_id, args.dest)
            print(f"{what}: {count} files checked", file=sys.stderr)
    except DownloadError as exc:
        print(f"download_artifacts: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
