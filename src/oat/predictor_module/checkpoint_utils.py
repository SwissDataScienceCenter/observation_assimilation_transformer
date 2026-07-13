from typing import Type, Tuple, Optional
from pathlib import Path

import pytorch_lightning as pl
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

    if predictor_cls is None:
        predictor_cls = _predictor_cls_from_config(cfg)

    predictor = predictor_cls.load_from_checkpoint(path, map_location=map_location)

    return predictor, cfg
