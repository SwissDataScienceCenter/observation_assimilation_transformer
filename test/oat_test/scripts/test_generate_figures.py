import argparse
import os
import re
import runpy
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
NAMES = [f"fig{i}" for i in range(1, 8)] + ["figA1", "figA2", "table1"]
EUROPE = [
    "era5-europe-mars-n320-2019-2024-1h-v1.zarr",
    "europe-static.zarr",
    "hourly_2019_2025_temp_prcp_wspd_wpgt_rhum.parquet",
    "station_quality.parquet",
]
SWISS = ["cosmo-1e_rechunked", "peakweather"]
COSMO_E1 = "mch-co1e-an-archive-0p01-2019-2024-1h-v1-pl13.zarr"


def _script(name):
    return REPO / "figures" / f"generate_{name}.sh"


def _roots(tmp_path, data=(), ckpts=False, bench=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    roots = {k: tmp_path / k for k in ("data", "ckpt", "bench")}
    for p in roots.values():
        p.mkdir()
    for name in data:
        (roots["data"] / name).mkdir()
    if ckpts:
        reg = yaml.safe_load((REPO / "configs" / "checkpoints.yaml").read_text())
        for models in reg.values():
            for key, value in models.items():
                if value and value != key:
                    path = roots["ckpt"] / value
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                    (path.parent / "config.yaml").touch()
    if bench:
        for label, suffix in [
            ("oat", ""),
            ("gp_oat", ""),
            ("gp_oat", "_exact"),
            ("oat_full_attention", ""),
        ]:
            for mode in ("fixed", "proportional"):
                name = f"2026-01-01_00-00-00_{label}_{mode}_B1{suffix}"
                (roots["bench"] / name).mkdir()
    return roots


def _run(name, roots):
    env = {
        **os.environ,
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "OAT_DATA_ROOT": str(roots["data"]),
        "OAT_CKPT_ROOT": str(roots["ckpt"]),
        "OAT_BENCH_DIR": str(roots["bench"]),
        "DRY_RUN": "1",
    }
    return subprocess.run(
        ["bash", str(_script(name))],
        cwd="/",
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


def _commands(stderr):
    return [
        shlex.split(line[2:])
        for line in stderr.splitlines()
        if line.startswith("+ python -m ")
    ]


def test_all_scripts_exist_and_parse():
    for name in NAMES:
        path = _script(name)
        assert path.is_file(), path
        text = path.read_text()
        assert text.startswith("#!/usr/bin/env bash\n"), path
        assert "#SBATCH" not in text, path
        subprocess.run(["bash", "-n", str(path)], check=True)


@pytest.mark.parametrize("name", NAMES)
def test_missing_inputs_stop_before_python(name, tmp_path):
    result = _run(name, _roots(tmp_path))
    assert result.returncode == 1, result.stderr
    assert "missing input" in result.stderr
    assert "Traceback" not in result.stderr
    assert not _commands(result.stderr)


@pytest.mark.parametrize("name", NAMES)
def test_dry_run_with_all_inputs(name, tmp_path):
    roots = _roots(tmp_path, [*EUROPE, *SWISS, COSMO_E1], ckpts=True, bench=True)
    result = _run(name, roots)
    assert result.returncode == 0, result.stderr
    commands = _commands(result.stderr)
    assert commands, result.stderr
    plots = [c for c in commands if c[2].startswith("scripts.plot_")]
    assert all("figures/out" in " ".join(c) for c in plots)


@pytest.mark.parametrize("name", ["fig4", "fig7"])
def test_mixed_figures_make_the_europe_part_first(name, tmp_path):
    result = _run(name, _roots(tmp_path, EUROPE, ckpts=True))
    assert result.returncode == 1
    assert "MeteoSwiss" in result.stderr
    europe = [c for c in _commands(result.stderr) if "era5_meteostat" in c]
    assert europe, result.stderr


@pytest.mark.parametrize("name", ["fig1", "fig2", "fig3", "fig6"])
def test_swiss_figures_stop_without_meteoswiss_data(name, tmp_path):
    result = _run(name, _roots(tmp_path, EUROPE, ckpts=True))
    assert result.returncode == 1
    assert "MeteoSwiss" in result.stderr
    assert not _commands(result.stderr)


def test_table1_uses_result_files_without_swiss_data(tmp_path):
    result = _run("table1", _roots(tmp_path, EUROPE, ckpts=True))
    assert result.returncode == 0, result.stderr
    commands = _commands(result.stderr)
    evaluated = {c[c.index("--dataset") + 1] for c in commands if "--dataset" in c}
    assert evaluated == {"era5_meteostat"}
    assert commands[-1][2] == "scripts.build_results_table"


def test_figA2_needs_the_figA1_runs(tmp_path):
    roots = _roots(tmp_path, ckpts=True)
    result = _run("figA2", roots)
    assert result.returncode == 1
    assert "Run figures/generate_figA1.sh first" in result.stderr


def test_every_named_generate_script_exists():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    named = set()
    for f in tracked:
        if f.startswith("docs/superpowers/") or not (REPO / f).is_file():
            continue
        text = (REPO / f).read_text(errors="ignore")
        named |= set(re.findall(r"generate_\w+\.sh", text))
    missing = [n for n in sorted(named) if not (REPO / "figures" / n).exists()]
    assert not missing


class _Parsed(Exception):
    pass


def test_every_command_matches_its_argparse_parser(tmp_path, monkeypatch):
    commands = []
    for name in NAMES:
        roots = _roots(tmp_path / name, [*EUROPE, *SWISS, COSMO_E1], True, True)
        commands += _commands(_run(name, roots).stderr)
    unique = {tuple(c[2:]) for c in commands if len(c) > 3}
    assert unique
    original = argparse.ArgumentParser.parse_args

    def parse_then_stop(self, args=None, namespace=None):
        original(self, args, namespace)  # SystemExit(2) on a bad option
        raise _Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_then_stop)
    monkeypatch.chdir(REPO)
    for module, *argv in sorted(unique):
        monkeypatch.setattr(sys, "argv", [module, *argv])
        with pytest.raises(_Parsed):
            runpy.run_module(module, run_name="__main__", alter_sys=False)
