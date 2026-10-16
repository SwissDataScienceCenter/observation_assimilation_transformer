"""Check the public tree for internal strings.

The check reads every tracked file (git ls-files) and every file in the
release stage folder. It prints each match with the file and the line.
It exits with code 1 when it finds a match. The patterns are in
release/leak_patterns.txt. The script uses only the standard library, so
CI can run it before the install. It does not check the git history.

Usage:
    python -m release.leak_check [--stage release/stage]
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PATTERN_FILE = Path(__file__).with_name("leak_patterns.txt")
_EXCLUDED = frozenset({"release/leak_check.py", "release/leak_patterns.txt"})


def load_pattern(path: Path = PATTERN_FILE) -> re.Pattern[bytes]:
    """Return one regex that matches any pattern line of ``path``.

    Empty lines and lines that start with ``#`` are ignored. Each line is
    put in its own group, so that ``|`` in a line keeps its meaning.
    """
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    parts = [ln for ln in lines if ln and not ln.startswith("#")]
    if not parts:
        raise ValueError(f"{path}: holds no pattern")
    return re.compile("|".join(f"(?:{p})" for p in parts).encode())


def tracked_files(root: Path = REPO_ROOT) -> list[Path]:
    """Return the tracked regular files of the repository at ``root``.

    The two files of the check are left out. A submodule (a folder) and a
    file that is deleted in the work tree are also left out.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    ).stdout.decode()
    names = [n for n in out.split("\0") if n and n not in _EXCLUDED]
    return [root / n for n in names if (root / n).is_file()]


def stage_files(stage: Path) -> list[Path]:
    """Return all files below ``stage`` in sorted order, or [] without it."""
    if not stage.is_dir():
        return []
    return sorted(p for p in stage.rglob("*") if p.is_file())


def find_leaks(files: list[Path], pattern: re.Pattern[bytes], root: Path) -> list[str]:
    """Return one ``<path>:<line>: <match>`` text for each match.

    The files are read as bytes, so binary files are checked too.
    """
    found: list[str] = []
    for path in files:
        data = path.read_bytes()
        name = path.relative_to(root) if path.is_relative_to(root) else path
        for m in pattern.finditer(data):
            line = data.count(b"\n", 0, m.start()) + 1
            text = m.group().decode(errors="replace")
            found.append(f"{name}:{line}: {text}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the tree for leaks.")
    parser.add_argument("--stage", type=Path, default=REPO_ROOT / "release" / "stage")
    args = parser.parse_args(argv)
    files = tracked_files() + stage_files(args.stage)
    leaks = find_leaks(files, load_pattern(), REPO_ROOT)
    for leak in leaks:
        print(leak)
    print(f"leak check: {len(files)} files, {len(leaks)} matches", file=sys.stderr)
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
