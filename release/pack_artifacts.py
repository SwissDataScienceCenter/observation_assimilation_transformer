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
    python -m release.pack_artifacts --what swiss --src <Swiss data root>

Then upload all files of release/upload/<what>/ to the Zenodo record.
"""

import argparse
import os
import shutil
import tarfile
from pathlib import Path

from oat.paths import REPO_ROOT, data_root
from release.leak_check import load_pattern
from scripts.download_artifacts import SUMS_NAME, file_digest

DATA_DIRS = ("era5-europe-mars-n320-2019-2024-1h-v1.zarr", "europe-static.zarr")
DATA_FILES = (
    "hourly_2019_2025_temp_prcp_wspd_wpgt_rhum.parquet",
    "station_quality.parquet",
)
SWISS_STORES = (
    "mch-co1e-an-archive-0p01-2019-2024-1h-v1-pl13.zarr",
    "cosmo-1e_rechunked/nwp_extracted.zarr",
    "cosmo-1e_rechunked/topo.zarr",
    "cosmo-1e_rechunked/nwp.zarr",
    "peakweather",
)
MAX_TAR_BYTES = 50 * 10**9
_ZARR_META = (".zattrs", ".zarray", ".zgroup", ".zmetadata")
SWISS_README = """# Swiss data pack (OAT)

This pack holds the COSMO-1E and PeakWeather data cropped to Switzerland.
It is for the reviewers of the paper.

## Content

- mch-co1e-an-archive-0p01-2019-2024-1h-v1-pl13.zarr: the metadata of the NWP
  archive (dates, grid, variable names and statistics). It holds no values.
- cosmo-1e_rechunked/nwp_extracted.zarr: the NWP values. Do not remove this
  store, because the dataset reads the values from it.
- cosmo-1e_rechunked/topo.zarr: the topography.
- cosmo-1e_rechunked/nwp.zarr: the x and y axes of the NWP grid.
- peakweather: the PeakWeather station data.

The data files are in tar files. A large store can be in more than one tar
file. SHA256SUMS lists the checksum of every packed file.

## Crop

The crop box is south 45.59, north 48.0, west 5.75 and east 10.81 degrees.
On the 786 x 1170 grid this gives rows 305 to 553 and columns 389 to 742.
This is 249 x 354 grid points.

## Variables

- NWP variables: T_G, T_2M, U_10M, V_10M, TOT_PREC_1H and CLCT.
- Topography variables: DEM, TPI_500M, TPI_2000M,
  WE_DERIVATIVE_2000M_SIGRATIO1 and SN_DERIVATIVE_2000M_SIGRATIO1.

## Not in this pack

The Swiss OOTG row, Figure 6 and the COSMO holdout row need the full
COSMO-1E domain. The full domain is not in this pack.

## Licence

- PeakWeather: CC BY 4.0, source: MeteoSwiss.
- COSMO-1E data: licence [LEGAL] - to be confirmed with MeteoSwiss before the
  pack is given out.

## Unpack

Put all tar files and SHA256SUMS in one folder. Then run this command from
the repository root:

    python -m scripts.unpack_swiss_data --src <folder> --dest <parent folder>

The data goes to <parent folder>/data. This is the data root (OAT_DATA_ROOT).
The command stops if a store exists already in <parent folder>/data.
If the SHA256 check fails, remove the listed stores below <parent folder>/data.
Then run the command again.
"""
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


def _tar_parts(
    src: Path, arc_root: str, stem: Path, max_bytes: int, sums: list[str]
) -> list[Path]:
    """Write the files below *src* to ``<stem>.tar`` or ``<stem>.NNN.tar``.

    A part is full when the next file would pass *max_bytes*. The count
    includes the tar headers and padding. A file that is larger than the
    budget causes a ValueError.
    """
    files = _walk(src)
    if not files:
        raise FileNotFoundError(f"{src}: no files to pack")
    block = tarfile.BLOCKSIZE
    budget = max_bytes - 2 * block - tarfile.RECORDSIZE  # end marker, record pad
    groups: list[list[Path]] = [[]]
    size = 0
    for f in files:
        # A pax header (2 blocks), the member header and the padded data.
        n = 3 * block + -(-f.stat().st_size // block) * block
        if n > budget:
            raise ValueError(f"{f}: larger than the part size {max_bytes} bytes")
        if groups[-1] and size + n > budget:
            groups.append([])
            size = 0
        groups[-1].append(f)
        size += n
    paths = [Path(f"{stem}.tar")]
    if len(groups) > 1:
        paths = [Path(f"{stem}.{i:03d}.tar") for i in range(len(groups))]
    for path, group in zip(paths, groups, strict=True):
        with tarfile.open(path, "w", dereference=True) as tar:
            for f in group:
                arc = f"{arc_root}/{f.relative_to(src).as_posix()}"
                tar.add(f, arcname=arc, recursive=False, filter=_scrub_owner)
                sums.append(f"{file_digest(f, 'sha256')}  {arc}")
    return paths


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


def _check_metadata(root: Path) -> None:
    """Raise ValueError if a zarr metadata file of a store holds a leak pattern."""
    pattern = load_pattern()
    for store in SWISS_STORES:
        for f in _walk(root / store):
            if f.name in _ZARR_META and pattern.search(f.read_bytes()):
                raise ValueError(f"{f}: holds an internal string")


def pack_swiss(root: Path, out: Path, max_bytes: int = MAX_TAR_BYTES) -> list[Path]:
    """Pack the Swiss data below *root*. Return the written files."""
    for store in SWISS_STORES:
        if not (root / store).is_dir():
            raise FileNotFoundError(f"{root / store}: not a folder")
    old = sorted(out.glob("*.tar"))
    if old:
        raise FileExistsError(f"{out}: holds tar files already, {old[0].name}")
    _check_metadata(root)
    out.mkdir(parents=True, exist_ok=True)
    sums: list[str] = []
    written: list[Path] = []
    for store in SWISS_STORES:
        stem = out / store.replace("/", "__")
        written += _tar_parts(root / store, f"data/{store}", stem, max_bytes, sums)
    readme = out / "README.md"
    readme.write_text(SWISS_README)
    return [*written, _write_sums(out, sums), readme]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Pack the release files for Zenodo.")
    p.add_argument("--what", choices=["checkpoints", "data", "swiss"], required=True)
    p.add_argument("--src", type=Path, default=None, help="Swiss data root.")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    out = args.out or REPO_ROOT / "release" / "upload" / args.what
    if args.what == "checkpoints":
        written = pack_checkpoints(_STAGE, out)
    elif args.what == "swiss":
        if args.src is None:
            p.error("--what swiss needs --src")
        written = pack_swiss(args.src, out)
    else:
        written = pack_data(data_root(), out)
    for path in written:
        print(f"{path}  {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
