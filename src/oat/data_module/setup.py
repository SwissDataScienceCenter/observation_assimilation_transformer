from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from oat.data_module import transform as T
from oat.data_module.data_split import create_splits as create_field_splits
from oat.data_module.field import Field
from oat.data_module.transform import NestedTransform, Transform

_GROUPS = ("observation_data", "query_data", "global_data")


def _instantiate_dataset(cfg: DictConfig) -> Dataset[Any]:
    dataset_type = cfg.dataset.get("dataset_type", "cosmo_peakweather")
    if dataset_type == "cosmo_peakweather":
        from oat.data_module.cosmo_peakweather_dataset import (
            CosmoPeakWeatherDataset as FieldCosmoPeakWeatherDataset,
        )

        return FieldCosmoPeakWeatherDataset(cfg.dataset)
    if dataset_type == "era5_meteostat":
        from oat.data_module.era5_meteostat_dataset import ERA5MeteostatDataset

        return ERA5MeteostatDataset(cfg.dataset)
    if dataset_type == "cosmo_meteostat":
        from oat.data_module.cosmo_meteostat_dataset import CosmoMeteostatDataset

        return CosmoMeteostatDataset(cfg.dataset)
    raise ValueError(
        f"Unknown dataset_type: {dataset_type!r}. "
        "Supported: 'cosmo_peakweather', 'era5_meteostat', 'cosmo_meteostat'."
    )


def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack per-sample fields into batched ``Field`` objects, group by group.

    ``global_data`` is optional: skip it when a sample has no such group.
    This keeps OAT and GP batches (no grid fields) unchanged.
    """
    out: dict[str, Any] = {"index": torch.tensor([s["index"] for s in samples])}
    for group in _GROUPS:
        if group not in samples[0]:
            continue
        names = list(samples[0][group]["fields"].keys())
        fields: dict[str, Field] = {}
        for name in names:
            flist = [s[group]["fields"][name] for s in samples]
            coords = torch.stack([f.coords for f in flist], dim=0)
            values = torch.stack([f.values for f in flist], dim=0)
            mask = (
                torch.stack([f.mask for f in flist], dim=0)
                if flist[0].mask is not None
                else None
            )
            fields[name] = Field(coords, values, list(flist[0].var_names), mask)
        out[group] = {"fields": fields}
    return out


def build_two_group_transform(
    inner_by_field: dict[str, T.Transform],
    grid_field_names: frozenset[str] = frozenset(),
) -> T.NestedTransform:
    """Wrap per-field inner transforms as independent FieldValueTransforms.

    Station fields (not in ``grid_field_names``) get one ``FieldValueTransform``
    per field under both ``observation_data`` and ``query_data``, built from a
    deep copy of the shared inner transform. This ensures the same
    ``nn.Module`` is never registered under two ``ModuleDict`` paths.

    Grid fields (named in ``grid_field_names``) go under a single
    ``global_data`` group instead. They are not duplicated in obs/query,
    since they share one grid, not per-station values.
    """
    station_inner = {
        n: t for n, t in inner_by_field.items() if n not in grid_field_names
    }
    grid_inner = {n: t for n, t in inner_by_field.items() if n in grid_field_names}

    def wrap(fields: dict[str, T.Transform]) -> dict[str, T.Transform]:
        return {
            n: T.FieldValueTransform(copy.deepcopy(inner))
            for n, inner in fields.items()
        }

    d: dict[str, Any] = {
        "observation_data": {"fields": wrap(station_inner)},
        "query_data": {"fields": wrap(station_inner)},
    }
    if grid_inner:
        d["global_data"] = {"fields": wrap(grid_inner)}
    return T.NestedTransform(d)


def validate_field_config_against_scalars(dataset: Any, cfg: DictConfig) -> None:
    """Check that field-config channel order matches the scalar var lists.

    Requires ``roles`` to define ``target`` and ``global``; ``geo_bias`` is
    only checked when present, so field configs without a geo field are
    tolerated.
    """
    fc = dataset.field_config
    roles = fc.roles
    for required in ("target", "global"):
        if required not in roles:
            raise ValueError(
                f"field roles must define {required!r}; got {sorted(roles)}"
            )
    checks = [("target", "obs_vars"), ("global", "nwp_vars")]
    if "geo_bias" in roles:
        checks.append(("geo_bias", "geo_vars"))
    if "global_grid" in roles:
        checks.append(("global_grid", "nwp_vars"))
    for role, scalar in checks:
        field_out = list(fc.fields[roles[role]].out)
        expected = [str(v) for v in cfg.dataset.get(scalar, [])]
        if role in ("global", "global_grid"):
            # The global field may append extra context channels (e.g. topo.*)
            # after the NWP scalars, so nwp_vars need only be a leading prefix
            # of the field's channels, not an exact match. Same rule applies
            # to the global_grid field.
            if field_out[: len(expected)] != expected:
                raise ValueError(
                    f"role {role!r} field {roles[role]!r} channels {field_out} "
                    f"must start with cfg.dataset.{scalar} {expected} (order matters)"
                )
        elif field_out != expected:
            raise ValueError(
                f"role {role!r} field {roles[role]!r} channels {field_out} "
                f"!= cfg.dataset.{scalar} {expected} (order matters)"
            )


def _setup_field_data_and_transform(cfg: DictConfig, logger: logging.Logger) -> tuple[
    Any,
    int,
    DataLoader[Any],
    DataLoader[Any],
    DataLoader[Any],
    T.NestedTransform,
    T.Transform,
]:
    """Field-path counterpart of ``setup_data_and_transform``."""
    dataset: Any = _instantiate_dataset(cfg)
    validate_field_config_against_scalars(dataset, cfg)
    logger.info(f"Dataset size: {len(dataset)} samples")

    splits = create_field_splits(dataset, **cfg.split, logger=logger)
    train_dataset, val_dataset, test_dataset = splits["datasets"]
    train_idx, val_idx, test_idx = splits["station_indices"]

    if all(
        cfg.split.get(k) is None
        for k in (
            "train_station_indices",
            "val_station_indices",
            "test_station_indices",
        )
    ):
        cfg.split.train_station_indices = train_idx
        cfg.split.val_station_indices = val_idx
        cfg.split.test_station_indices = test_idx

    # Stats over TRAIN stations, restricted to train/val time steps.
    inner_by_field, output_transform = dataset.compute_field_leaves(
        num_data_points=cfg.dataset.data_points_for_stats,
        data_indices=train_dataset._time_indices,
        target_point_indices=np.asarray(train_idx),
        type=cfg.model.get("transform_type", "real"),
    )
    target_cs = dataset.target_coord_source()
    fcs = dataset.field_coord_sources()
    grid_field_names = frozenset(n for n, cs in fcs.items() if cs != target_cs)
    input_transform = build_two_group_transform(inner_by_field, grid_field_names)

    train_loader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate, **cfg.dataloader.train
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate,
        **cfg.dataloader.validation,
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, collate_fn=collate, **cfg.dataloader.test
    )
    return (
        dataset,
        len(train_dataset),
        train_loader,
        val_loader,
        test_loader,
        input_transform,
        output_transform,
    )


def setup_data_and_transform(cfg: DictConfig, logger: logging.Logger) -> tuple[
    Any,
    int,
    DataLoader,
    DataLoader,
    DataLoader,
    NestedTransform,
    Transform,
]:
    """Build the field dataset, the splits, the loaders and the transforms."""
    return _setup_field_data_and_transform(cfg, logger)
