"""Pack the release files for the two Zenodo records.

Zenodo keeps a flat list of files, so each folder goes into a tar file. The
tar member names are the paths below --dest of scripts/download_artifacts.py.
SHA256SUMS lists every packed file with the same path. The tar files are not
compressed: the checkpoints are zip files and the zarr chunks are compressed
already.

    checkpoints record: checkpoints.tar   checkpoints/<family>/<run>/...
                        results.tar       data/results/<file>
                        SHA256SUMS
    data record:        <name>.zarr.tar   data/<name>.zarr/...  (one per zarr)
                        <name>.parquet    data/<name>.parquet
                        SHA256SUMS

Usage (from the repository root):
    python -m release.scrub_checkpoints
    python -m release.pack_artifacts --what checkpoints
    OAT_DATA_ROOT=<data root> python -m release.pack_artifacts --what data

Then upload all files of release/upload/<what>/ to the Zenodo record.
"""

import argparse
import os
import shutil
import tarfile
from pathlib import Path

from oat.paths import REPO_ROOT, data_root
from scripts.download_artifacts import SUMS_NAME, file_digest

DATA_DIRS = ("era5-europe-mars-n320-2019-2024-1h-v1.zarr", "europe-static.zarr")
DATA_FILES = (
    "hourly_2019_2025_temp_prcp_wspd_wpgt_rhum.parquet",
    "station_quality.parquet",
)
_STAGE = REPO_ROOT / "release" / "stage"


def _walk(src: Path) -> list[Path]:
    """Return all files below *src* in sorted order. Follow symbolic links."""
    found: list[Path] = []
    for root, _dirs, files in os.walk(src, followlinks=True):
        found += [Path(root) / f for f in files]
    return sorted(found)


def _scrub_owner(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Remove the owner names and ids. Keep the mode and the time."""
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _tar_tree(src: Path, arc_root: str, tar_path: Path, sums: list[str]) -> None:
    """Write the files below *src* to *tar_path* as ``<arc_root>/<rel>``."""
    files = _walk(src)
    if not files:
        raise FileNotFoundError(f"{src}: no files to pack")
    with tarfile.open(tar_path, "w", dereference=True) as tar:
        for f in files:
            arc = f"{arc_root}/{f.relative_to(src).as_posix()}"
            tar.add(f, arcname=arc, recursive=False, filter=_scrub_owner)
            sums.append(f"{file_digest(f, 'sha256')}  {arc}")


def _write_sums(out: Path, sums: list[str]) -> Path:
    path = out / SUMS_NAME
    path.write_text(
        "\n".join(sorted(sums, key=lambda s: s.split(maxsplit=1)[1])) + "\n"
    )
    return path


def pack_checkpoints(stage: Path, out: Path) -> list[Path]:
    """Pack the staged checkpoints and result files. Return the written files."""
    out.mkdir(parents=True, exist_ok=True)
    sums: list[str] = []
    _tar_tree(stage / "checkpoints", "checkpoints", out / "checkpoints.tar", sums)
    _tar_tree(stage / "results", "data/results", out / "results.tar", sums)
    return [out / "checkpoints.tar", out / "results.tar", _write_sums(out, sums)]


def pack_data(root: Path, out: Path) -> list[Path]:
    """Pack the Europe dataset below *root*. Return the written files."""
    for name in DATA_DIRS:
        if not (root / name).is_dir():
            raise FileNotFoundError(f"{root / name}: not a folder")
    for name in DATA_FILES:
        if not (root / name).is_file():
            raise FileNotFoundError(f"{root / name}: not a file")
    out.mkdir(parents=True, exist_ok=True)
    sums: list[str] = []
    written: list[Path] = []
    for name in DATA_DIRS:
        _tar_tree(root / name, f"data/{name}", out / f"{name}.tar", sums)
        written.append(out / f"{name}.tar")
    for name in DATA_FILES:
        shutil.copyfile(root / name, out / name)  # follows a symbolic link
        sums.append(f"{file_digest(out / name, 'sha256')}  data/{name}")
        written.append(out / name)
    return [*written, _write_sums(out, sums)]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Pack the release files for Zenodo.")
    p.add_argument("--what", choices=["checkpoints", "data"], required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    out = args.out or REPO_ROOT / "release" / "upload" / args.what
    if args.what == "checkpoints":
        written = pack_checkpoints(_STAGE, out)
    else:
        written = pack_data(data_root(), out)
    for path in written:
        print(f"{path}  {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
