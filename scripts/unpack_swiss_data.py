"""Unpack the Swiss data pack with this command.

The folder given with --src holds the tar files and the SHA256SUMS file of
the pack. The tar member names start with ``data/``. So the data goes to
<dest>/data, and <dest>/data is the data root (OAT_DATA_ROOT). The command
stops before it writes a file if SHA256SUMS is missing, if a tar member is
not in SHA256SUMS, or if a store exists already below <dest>/data.
It checks every file with SHA256SUMS after the unpack. It keeps the tar files.

Usage (from the repository root):
    python -m scripts.unpack_swiss_data --src <folder> --dest <parent folder>
"""

import argparse
import sys
import tarfile
from pathlib import Path

from scripts.download_artifacts import (
    SUMS_NAME,
    DownloadError,
    _check_member,
    verify_sums,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Unpack the Swiss data pack.")
    p.add_argument("--src", type=Path, required=True, help="Folder with the tar files.")
    p.add_argument("--dest", type=Path, required=True, help="Parent of the data root.")
    args = p.parse_args(argv)
    sums = args.src / SUMS_NAME
    if not sums.is_file():
        raise DownloadError(f"{args.src}: no {SUMS_NAME} file")
    listed = {
        fields[1]
        for fields in (ln.split(maxsplit=1) for ln in sums.read_text().splitlines())
        if len(fields) == 2
    }
    tars = sorted(args.src.glob("*.tar"))
    if not tars:
        raise DownloadError(f"{args.src}: no tar files")
    for path in tars:
        try:
            with tarfile.open(path) as tar:
                for member in tar.getmembers():
                    _check_member(path.name, member)
                    if member.name not in listed:
                        raise DownloadError(
                            f"{path.name}: {member.name} is not in {SUMS_NAME}"
                        )
                    # Check the first two name parts, for example
                    # data/cosmo-1e_rechunked. A tar lists only files.
                    target = args.dest.joinpath(*Path(member.name).parts[:2])
                    if target.exists():
                        raise DownloadError(f"{target} exists")
        except tarfile.TarError as exc:
            raise DownloadError(f"{path.name}: {exc}") from exc
    for path in tars:
        try:
            with tarfile.open(path) as tar:
                tar.extractall(args.dest, filter="data")
        except tarfile.TarError as exc:
            raise DownloadError(f"{path.name}: {exc}") from exc
    count = verify_sums(sums, args.dest)
    print(f"{count} files checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
