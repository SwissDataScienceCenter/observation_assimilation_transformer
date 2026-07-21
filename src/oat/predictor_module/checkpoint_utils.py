from typing import Type, Tuple, Optional
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, DictConfig

_DEFAULT_REGISTRY = Path(__file__).parents[3] / "configs" / "checkpoints.yaml"

# Maps substrings of model_cls to the predictor class that loads them.
_MODEL_CLS_TO_PREDICTOR = {
    "GPBaseline": "oat.predictor_module.gp_predictor.GPPredictor",
    "NWPInterpolator": "oat.predictor_module.baseline_predictor.NWPInterpolatorPredictor",
}
_DEFAULT_PREDICTOR = "oat.predictor_module.oat_predictor.OATPredictor"


def _predictor_cls_from_config(cfg: DictConfig) -> Type[pl.LightningModule]:
    """Infer predictor class from cfg.model.model_cls."""
    import importlib
    model_cls_str = cfg.model.model_cls
    target = _DEFAULT_PREDICTOR
    for fragment, predictor_path in _MODEL_CLS_TO_PREDICTOR.items():
        if fragment in model_cls_str:
            target = predictor_path
            break
    module_path, cls_name = target.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), cls_name)


def _load_ootg_predictor(
    path: Path,
    cfg: DictConfig,
    map_location: str | None,
) -> Tuple[pl.LightningModule, DictConfig]:
    """Load an OOTGPredictor from an OOTG checkpoint directory."""
    from oat.baselines.coord_utils import lonlat_swap, lv95_to_latlon
    from oat.baselines.ootg_model import build_ootg_model
    from oat.baselines.ootg_predictor import OOTGPredictor
    from oat.data_module.transform import StandardNormalizeTransform, Transform

    stats = torch.load(
        path.parent / "ootg_stats.pt", map_location=map_location, weights_only=False
    )

    model = build_ootg_model(**stats["model_kwargs"])
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model_state = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(model_state)

    if "output_transform" in stats and stats["output_transform"] is not None:
        output_transform = Transform.from_dict(stats["output_transform"])
    else:
        # Backward compat: reconstruct from obs_mean/obs_std (Normal vars only)
        obs_mean = stats["obs_mean"]
        obs_std = stats["obs_std"]
        dist_names = stats["dist_names"]
        mean_t = obs_mean.clone()
        std_t = obs_std.clone()
        for v, dn in enumerate(dist_names):
            if dn != "normal":
                mean_t[v] = 0.0
                std_t[v] = 1.0
        output_transform = StandardNormalizeTransform(mean_t, std_t, channel_dim=-1)

    coord_fn = (
        lv95_to_latlon if stats["coord_fn_name"] == "lv95_to_latlon" else lonlat_swap
    )

    use_topo = stats["model_kwargs"].get("use_topo", False)
    topo_transform = None
    if use_topo and stats.get("topo_transform") is not None:
        topo_transform = Transform.from_dict(stats["topo_transform"])

    predictor = OOTGPredictor(
        model=model,
        obs_vars=stats["obs_vars"],
        obs_mean=stats["obs_mean"],
        obs_std=stats["obs_std"],
        nwp_mean=stats["nwp_mean"],
        nwp_std=stats["nwp_std"],
        xc_grid_base=stats["xc_grid_base"],
        coord_to_latlon=coord_fn,
        dist_names=stats["dist_names"],
        n_nwp=stats["n_nwp"],
        nwp_stride=stats["nwp_stride"],
        output_transform=output_transform,
        use_topo=use_topo,
        topo_transform=topo_transform,
    )
    return predictor, cfg


def _load_gp_predictor(
    path: Path,
    cfg: DictConfig,
    map_location: str | None,
) -> Tuple[pl.LightningModule, DictConfig]:
    """Load a GPPredictor from a GP checkpoint.

    ``GPPredictor.__init__`` needs the (weightless) GP model and the
    input/output transforms, none of which are stored as reconstructable
    hyperparameters, so ``load_from_checkpoint`` cannot rebuild it. Instead we
    reconstruct the predictor from the config + dataset and then restore every
    weight from the checkpoint. The transforms are ``nn.Module`` buffers saved
    in the state_dict, so we only need correctly-shaped transforms here — the
    fitted values are overwritten by ``load_state_dict`` — which lets us skip the
    (slow) normalization-stats gathering.
    """
    import logging

    from oat.data_module.setup import setup_data_and_transform
    from oat.predictor_module.gp_predictor import setup_gp_predictor

    logger = logging.getLogger(__name__)
    cfg.dataset.data_points_for_stats = 0
    dataset, _, _, _, _, input_transform, output_transform = setup_data_and_transform(
        cfg, logger
    )

    predictor = setup_gp_predictor(
        cfg=cfg,
        field_extent={k: v.tolist() for k, v in dataset.nwp_extent.items()},
        obs_vars=list(dataset.config.obs_vars),
        transforms=(input_transform, output_transform),
    )
    state = torch.load(path, map_location=map_location, weights_only=False)
    predictor.load_state_dict(state["state_dict"])
    return predictor, cfg


def resolve_checkpoint(
    dataset: str,
    model: str,
    registry_path: str | Path | None = None,
) -> str:
    """Return the checkpoint path for (dataset, model) from configs/checkpoints.yaml."""
    path = Path(registry_path) if registry_path is not None else _DEFAULT_REGISTRY
    registry = OmegaConf.load(path)
    try:
        value = registry[dataset][model]
    except KeyError:
        raise KeyError(
            f"No checkpoint registered for dataset={dataset!r}, model={model!r}"
        )
    if value is None:
        raise ValueError(
            f"Checkpoint for dataset={dataset!r}, model={model!r} is not set (null) in {path}"
        )
    return str(value)


def load_checkpoint(
    predictor_cls: Optional[Type[pl.LightningModule]],
    checkpoint_file_path: str,
    map_location: str | None = None,
) -> Tuple[pl.LightningModule, DictConfig]:
    """
    Load and return the Predictor and its config from a checkpoint file.

    predictor_cls: predictor class to use. When None, auto-detected from
      cfg.model.model_cls in the checkpoint's sibling config.yaml.
    checkpoint_file_path: path to the .ckpt file (not the folder).
    map_location: passed to torch.load (e.g. "cpu" to force CPU-only loading).
    """
    path = Path(checkpoint_file_path)
    config_path = path.parent / "config.yaml"

    cfg = OmegaConf.load(config_path)

    if (path.parent / "ootg_stats.pt").exists():
        return _load_ootg_predictor(path, cfg, map_location)

    if "GPBaseline" in str((cfg.get("model") or {}).get("model_cls", "")):
        return _load_gp_predictor(path, cfg, map_location)

    if predictor_cls is None:
        predictor_cls = _predictor_cls_from_config(cfg)

    predictor = predictor_cls.load_from_checkpoint(path, map_location=map_location)

    return predictor, cfg
