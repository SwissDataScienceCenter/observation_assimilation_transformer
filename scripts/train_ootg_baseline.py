"""Train the OOTG (gridded-TNP) baseline via the field data path.

Usage:
    python -m scripts.train_ootg_baseline dataset=cosmo_peakweather/temp_ootg
    python -m scripts.train_ootg_baseline dataset=cosmo_peakweather/temp_precip_ootg
"""

import logging
import os
from datetime import datetime

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from oat.baselines.ootg_model import build_ootg_model
from oat.baselines.ootg_predictor import create_ootg_predictor
from oat.data_module.setup import _setup_field_data_and_transform

_LOG = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="ootg_baseline", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Seed before the split, so the random station partition is identical
    # across runs. Default 42 matches OAT (configs/config.yaml), so OOTG and
    # OAT share the split.
    pl.seed_everything(cfg.get("seed", 42))
    torch.set_float32_matmul_precision("high")

    (
        dataset,
        _train_size,
        train_loader,
        val_loader,
        test_loader,
        input_transform,
        output_transform,
    ) = _setup_field_data_and_transform(cfg, _LOG)

    roles = dataset.roles()
    grid_field = roles["global_grid"]
    grid_cs = dataset.field_coord_sources()[grid_field]
    native_shape = dataset.grid_shape(grid_cs)
    if native_shape is None:
        raise ValueError(
            f"No grid shape for coord source {grid_cs!r}; "
            "OOTG needs a rectangular NWP grid (ERA5 N320 is not supported)."
        )
    ablate_obs = cfg.get("ablate_obs", False)
    ablate_grid = cfg.get("ablate_grid", False)
    if ablate_obs and ablate_grid:
        raise ValueError("ablate_obs and ablate_grid together leave no signal.")
    if ablate_obs or ablate_grid:
        _LOG.info("ABLATION active: ablate_obs=%s ablate_grid=%s", ablate_obs, ablate_grid)

    _DEFAULT_STRIDES = {"cosmo_peakweather": 10, "era5_meteostat": 4}
    dataset_type = cfg.dataset.dataset_type
    if dataset_type not in _DEFAULT_STRIDES:
        raise ValueError(f"Unknown dataset_type for OOTG: {dataset_type!r}")
    nwp_stride = cfg.get("nwp_stride", None) or _DEFAULT_STRIDES[dataset_type]
    # Ceiling division matches the ::stride slice in _unwrap_fields_to_ootg_batch.
    strided_shape = (
        -(-native_shape[0] // nwp_stride),
        -(-native_shape[1] // nwp_stride),
    )
    _LOG.info(
        "Grid %s -> stride %d -> pseudo-token grid %s",
        native_shape,
        nwp_stride,
        strided_shape,
    )

    n_nwp = len(cfg.dataset.nwp_vars)
    dim_y = len(cfg.dataset.obs_vars)
    v2d = dict(cfg.dataset.obs_variable_to_distribution) if cfg.dataset.get("obs_variable_to_distribution") else None
    model = build_ootg_model(
        grid_shape=strided_shape,
        dim_gridded_y=n_nwp,
        dim_y=dim_y,
        obs_variable_to_distribution=v2d,
        use_topo=cfg.get("use_topo", False),
    )
    _LOG.info("Parameters: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    predictor = create_ootg_predictor(
        model=model,
        dataset=dataset,
        output_transform=output_transform,
        input_transform=input_transform,
        cfg=cfg,
    )

    now_str = datetime.now().strftime("%Y%m%d")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    ablation_tag = (
        "_nwponly"
        if cfg.get("ablate_obs", False)
        else "_obsonly"
        if cfg.get("ablate_grid", False)
        else ""
    )
    dataset_tag = f"{cfg.dataset.dataset_type}_{'_'.join(cfg.dataset.obs_vars)}"
    default_run_name = (
        f"ootg-baseline{ablation_tag}_{dataset_tag}_{now_str}_job{job_id}"
    )
    run_name = cfg.get("run_name") or default_run_name
    checkpoint_dir = os.path.join("checkpoints", cfg.logging.group, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Persist the resolved config (the split indices are written back by
    # _setup_field_data_and_transform) and the model kwargs, so
    # evaluate_test_set.py can rebuild this checkpoint from files alone.
    # Guard to rank 0: torch.save is not atomic, and all ranks have
    # identical content.
    if int(os.environ.get("SLURM_PROCID", 0)) == 0:
        OmegaConf.save(cfg, os.path.join(checkpoint_dir, "config.yaml"))
        torch.save(
            {
                "model_kwargs": {
                    "dim_y": dim_y,
                    "dim_gridded_y": n_nwp,
                    "obs_variable_to_distribution": v2d,
                    "grid_shape": strided_shape,
                    "use_topo": cfg.get("use_topo", False),
                },
            },
            os.path.join(checkpoint_dir, "ootg_stats.pt"),
        )
        _LOG.info("Saved config.yaml and ootg_stats.pt to %s", checkpoint_dir)

    wandb_logger = WandbLogger(
        name=run_name,
        project=cfg.logging.project,
        group=cfg.logging.group,
        entity=cfg.logging.entity,
        log_model=False,
        settings=wandb.Settings(init_timeout=300),
    )
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch={epoch}-val_loss={val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(),
    ]

    trainer = pl.Trainer(
        **cfg.get("trainer", {}),
        callbacks=callbacks,
        logger=wandb_logger,
    )
    trainer.fit(predictor, train_loader, val_loader)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
