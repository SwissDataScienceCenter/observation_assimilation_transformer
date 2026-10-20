"""Write a release copy of the registered checkpoints.

For each entry of configs/checkpoints.yaml, this script copies the best
.ckpt, the config.yaml (and ootg_stats.pt for OOTG) to the stage folder.
It rewrites the data paths to ${paths.data_root}/<file>, removes the
W&B entity and group, and removes internal paths from the .ckpt. It also
copies the result files of the paper table. It writes SHA256SUMS last.
It fails on a missing file or on an internal path in a staged file.
The original files do not change.

Usage:
    python -m release.scrub_checkpoints
"""

import argparse
import dataclasses
import hashlib
import shutil
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from oat.paths import REGISTRY_PATH, REPO_ROOT, checkpoint_root
from release.leak_check import load_pattern

LEAK_RE = load_pattern()
_DATA_KEYS = (
    "nwp_path",
    "topo_path",
    "obs_path",
    "station_meta_path",
    "nwp_extracted_path",
    "ref_nwp_path",
    "cache_dir",
)
_OOTG_KEYS = ("dataloader", "split.train_station_indices", "model.transform_type")
_SPLIT_SUFFIXES = ("", "_spatial_region_holdout")


class ScrubError(RuntimeError):
    """A checkpoint cannot be released as it is."""


def _is_ootg(cfg: DictConfig) -> bool:
    # Only the OOTG training schema has the top-level key ``use_topo``.
    return "use_topo" in cfg


def dataset_base(dataset_type: str, ootg: bool) -> DictConfig:
    """Return the tracked base config of a dataset type.

    OOTG runs read the full COSMO-1E domain. For them, merge the
    full_domain.yaml of the same folder when that file exists.
    """
    folder = REPO_ROOT / "configs/dataset" / dataset_type
    base = OmegaConf.load(folder / "base.yaml")
    full = folder / "full_domain.yaml"
    if ootg and full.is_file():
        base = OmegaConf.merge(base, OmegaConf.load(full))
    assert isinstance(base, DictConfig)
    return base


def _absolute_strings(node: Any, prefix: str = "") -> list[str]:
    """Return the keys of all string values that start with ``/``."""
    found: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            found += _absolute_strings(v, f"{prefix}{k}.")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            found += _absolute_strings(v, f"{prefix}{i}.")
    elif isinstance(node, str) and node.startswith("/"):
        found.append(prefix.rstrip("."))
    return found


def scrub_config(cfg: DictConfig, dataset_base: DictConfig) -> DictConfig:
    """Return a copy of a run config without internal paths.

    ``dataset_base`` is the tracked configs/dataset/<type>/base.yaml. Its
    unresolved ``${paths.data_root}/<file>`` values replace the data paths.
    A data key that is not in ``dataset_base`` is removed (legacy keys).
    """
    raw = OmegaConf.to_container(cfg, resolve=False)
    base = OmegaConf.to_container(dataset_base, resolve=False)
    assert isinstance(raw, dict) and isinstance(base, dict)
    for key in _DATA_KEYS:
        if key in raw["dataset"]:
            if key in base:
                raw["dataset"][key] = base[key]
            else:
                del raw["dataset"][key]
    if "logging" in raw:
        raw["logging"]["entity"] = None
        raw["logging"]["group"] = None
    if "resume_from_checkpoint" in raw:
        raw["resume_from_checkpoint"] = None
    raw["paths"] = {"data_root": "${oc.env:OAT_DATA_ROOT,data}"}
    left = _absolute_strings(raw)
    if left:
        raise ScrubError(f"Absolute paths remain at keys: {left}")
    out = OmegaConf.create(raw)
    if _is_ootg(out):
        for key in _OOTG_KEYS:
            if OmegaConf.select(out, key) is None:
                raise ScrubError(f"OOTG config misses the key {key!r}")
    return out


def _detach(obj: Any) -> Any:
    """Replace OmegaConf nodes with copies that have no parent config."""
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.create(OmegaConf.to_container(obj, resolve=True))
    if isinstance(obj, dict):
        return {k: _detach(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_detach(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_detach(v) for v in obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            object.__setattr__(obj, f.name, _detach(getattr(obj, f.name)))
    return obj


def scrub_ckpt(ckpt: dict[str, Any]) -> dict[str, Any]:
    """Remove the callback state and detach the hyperparameter nodes."""
    out = dict(ckpt)
    out.pop("callbacks", None)
    if out.get("hyper_parameters") is not None:
        out["hyper_parameters"] = _detach(out["hyper_parameters"])
    return out


def check_no_leak(path: Path) -> None:
    """Raise ScrubError when the file holds a leak pattern."""
    match = LEAK_RE.search(path.read_bytes())
    if match:
        raise ScrubError(f"{path}: holds {match.group().decode()!r}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _copy_results(
    registry: dict[str, Any], results_dir: Path, stage: Path
) -> list[Path]:
    from scripts.build_results_table import _resolve_tag

    out_dir = stage / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for dataset, models in registry.items():
        for key, value in (models or {}).items():
            tag = _resolve_tag(dataset, key, value)
            if tag is None:
                continue
            found = False
            for suffix in _SPLIT_SUFFIXES:
                for ext in (".csv", ".summary.json"):
                    src = results_dir / f"{tag}{suffix}_metrics_test{ext}"
                    if src.is_file():
                        shutil.copy2(src, out_dir / src.name)
                        written.append(out_dir / src.name)
                        found = True
            if not found:
                raise ScrubError(f"No test results for {dataset}/{key} ({tag})")
    return written


def scrub_all(
    registry_path: Path, src_root: Path, stage: Path, results_dir: Path
) -> list[Path]:
    """Stage all registered checkpoints and results. Return the staged files."""
    loaded = OmegaConf.to_container(OmegaConf.load(registry_path))
    assert isinstance(loaded, dict)
    registry: dict[str, Any] = {str(k): v for k, v in loaded.items()}
    written: list[Path] = []
    for dataset, models in registry.items():
        for key, value in (models or {}).items():
            if value is None or value == key:
                continue
            rel = Path(str(value))
            if rel.is_absolute():
                raise ScrubError(f"{dataset}/{key}: absolute registry value {rel}")
            src = src_root / rel
            if not src.is_file():
                raise ScrubError(f"{dataset}/{key}: {src} is not on disk")
            dst = stage / "checkpoints" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)

            cfg = OmegaConf.load(src.parent / "config.yaml")
            assert isinstance(cfg, DictConfig)
            base = dataset_base(cfg.dataset.dataset_type, _is_ootg(cfg))
            OmegaConf.save(scrub_config(cfg, base), dst.parent / "config.yaml")
            written.append(dst.parent / "config.yaml")

            ckpt = torch.load(src, map_location="cpu", weights_only=False)
            torch.save(scrub_ckpt(ckpt), dst)
            written.append(dst)

            stats = src.parent / "ootg_stats.pt"
            if stats.is_file():
                shutil.copy2(stats, dst.parent / stats.name)
                written.append(dst.parent / stats.name)
    written += _copy_results(registry, results_dir, stage)

    for path in written:
        check_no_leak(path)
    lines = [f"{_sha256(p)}  {p.relative_to(stage)}" for p in sorted(written)]
    (stage / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=Path, default=REPO_ROOT / "release/stage")
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "data/results")
    args = p.parse_args()
    if args.stage.exists():
        raise SystemExit(f"{args.stage} exists. Remove it first.")
    files = scrub_all(REGISTRY_PATH, checkpoint_root(), args.stage, args.results_dir)
    print(f"Staged {len(files)} files in {args.stage}")


if __name__ == "__main__":
    main()
