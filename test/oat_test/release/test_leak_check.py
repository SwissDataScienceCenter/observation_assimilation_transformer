import subprocess

import pytest

from release.leak_check import (
    find_leaks,
    load_pattern,
    stage_files,
    tracked_files,
)

# Each sample is split into two literals, so this file holds no pattern.
_SHIPPED_HITS = [
    b"/scr" b"atch/mch/x",
    b"/us" b"ers/me",
    b"/cap" b"stor/a",
    b"/sto" b"re/a",
    b"/sto" b"re_new/a",
    b"me@meteo" b"swiss.ch",
    b"entity: sdsc" b"-mch",
]
_SHIPPED_MISSES = [b"scratch_pad", b"me@meteoswiss-ch", b"sdsc_mch", b"users"]


def test_load_pattern_skips_comments_and_blank_lines(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("# comment\n\nfoo\nba+r\n")
    pat = load_pattern(f)
    assert pat.search(b"xx foo") and pat.search(b"baaar")
    assert not pat.search(b"comment")


def test_load_pattern_keeps_each_line_grouped(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("a|b\n^c$\n")
    pat = load_pattern(f)
    assert pat.fullmatch(b"a") and pat.fullmatch(b"b") and pat.fullmatch(b"c")
    assert not pat.fullmatch(b"bc")


def test_load_pattern_fails_on_empty_file(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("# only a comment\n")
    with pytest.raises(ValueError, match="no pattern"):
        load_pattern(f)


@pytest.mark.parametrize("sample", _SHIPPED_HITS)
def test_shipped_patterns_match(sample):
    assert load_pattern().search(sample)


@pytest.mark.parametrize("sample", _SHIPPED_MISSES)
def test_shipped_patterns_skip_near_misses(sample):
    assert not load_pattern().search(sample)


def _pattern(tmp_path, line):
    f = tmp_path / "pat.txt"
    f.write_text(line + "\n")
    return load_pattern(f)


def test_find_leaks_reports_file_and_line(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("ok\nSECRET here\nok\nSECRET\n")
    b = tmp_path / "b.bin"
    b.write_bytes(b"\xff\x00\nxx\xfeSECRET\x00")
    clean = tmp_path / "c.txt"
    clean.write_text("nothing\n")
    leaks = find_leaks([a, b, clean], _pattern(tmp_path, "SECRET"), tmp_path)
    assert leaks == ["a.txt:2: SECRET", "a.txt:4: SECRET", "b.bin:2: SECRET"]


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_tracked_files_skips_excluded_missing_and_folders(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "release").mkdir()
    for name in (
        "keep.txt",
        "gone.txt",
        "release/leak_check.py",
        "release/leak_patterns.txt",
    ):
        (tmp_path / name).write_text("x\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    _git(sub, "init", "-q")
    (sub / "f.txt").write_text("x\n")
    _git(sub, "add", "f.txt")
    _git(sub, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    _git(tmp_path, "add", ".")  # sub is added as a gitlink (a folder)
    (tmp_path / "gone.txt").unlink()
    assert tracked_files(tmp_path) == [tmp_path / "keep.txt"]


def test_stage_files(tmp_path):
    assert stage_files(tmp_path / "missing") == []
    (tmp_path / "s" / "d").mkdir(parents=True)
    (tmp_path / "s" / "d" / "x.ckpt").write_bytes(b"x")
    (tmp_path / "s" / "y.csv").write_text("y")
    assert stage_files(tmp_path / "s") == [
        tmp_path / "s" / "d" / "x.ckpt",
        tmp_path / "s" / "y.csv",
    ]
