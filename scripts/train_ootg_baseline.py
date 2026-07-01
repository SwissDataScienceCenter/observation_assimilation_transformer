"""Train OOTG Gridded-TNP baseline on COSMO-1E+PeakWeather or ERA5+Meteostat.

Usage:
    python -m scripts.train_ootg_baseline dataset=cosmo_peakweather/temp
    python -m scripts.train_ootg_baseline dataset=cosmo_peakweather/temp_precip
    python -m scripts.train_ootg_baseline dataset=era5_meteostat/temp
    python -m scripts.train_ootg_baseline dataset=era5_meteostat/temp_precip epochs=10
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
from torch.utils.data import DataLoader

from oat.baselines.ootg_generator import compute_norm_stats, make_ootg_generator
from oat.baselines.ootg_model import build_ootg_model
from oat.baselines.ootg_predictor import OOTGPredictor
from oat.data_module.data_split import create_splits
from oat.data_module.setup import _instantiate_dataset

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="ootg_baseline", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    log.info("Loading dataset...")
    full_ds = _instantiate_dataset(cfg)
    splits = create_splits(full_ds, **cfg.split, logger=log)
    train_wrap, val_wrap, _ = splits["datasets"]

    obs_vars = list(full_ds.config.obs_vars)
    nwp_vars = list(full_ds.config.nwp_vars)
    topo_vars = list(full_ds.config.topo_vars)
    v2d = dict(full_ds.config.obs_variable_to_distribution)
    dim_y, dim_gridded_y = len(obs_vars), len(nwp_vars) + len(topo_vars)
    log.info(
        "obs_vars=%s  v2d=%s  nwp_vars=%s  topo_vars=%s",
        obs_vars,
        v2d,
        nwp_vars,
        topo_vars,
    )

    log.info("Computing normalization stats (n=%d)...", cfg.get("norm_samples", 500))
    obs_mean, obs_std, nwp_mean, nwp_std = compute_norm_stats(
        full_ds,
        n_samples=cfg.get("norm_samples", 500),
        time_indices=train_wrap._time_indices,
    )
    log.info("obs_mean=%s  obs_std=%s", obs_mean.tolist(), obs_std.tolist())

    _, output_transform = full_ds.get_transform(
        num_data_points=cfg.get("norm_samples", 500),
        data_indices=train_wrap._time_indices,
        type=cfg.get("transform_type", "keep_support"),
    )

    gen_train = make_ootg_generator(
        full_ds,
        obs_mean=obs_mean,
        obs_std=obs_std,
        nwp_mean=nwp_mean,
        nwp_std=nwp_std,
        nwp_stride=cfg.get("nwp_stride", None),
        time_indices=train_wrap._time_indices,
        output_transform=output_transform,
        pc_context=cfg.get("pc_context", 0.8),
        samples_per_epoch=cfg.get("samples_per_epoch", 10000),
    )
    gen_val = make_ootg_generator(
        full_ds,
        obs_mean=obs_mean,
        obs_std=obs_std,
        nwp_mean=nwp_mean,
        nwp_std=nwp_std,
        nwp_stride=cfg.get("nwp_stride", None),
        time_indices=val_wrap._time_indices,
        output_transform=output_transform,
        samples_per_epoch=cfg.get("val_samples_per_epoch", 1000),
    )

    model = build_ootg_model(
        dim_y=dim_y,
        dim_gridded_y=dim_gridded_y,
        obs_variable_to_distribution=v2d,
        embed_dim=cfg.get("embed_dim", 128),
        x_embed_dim=cfg.get("x_embed_dim", 128),
        num_heads=cfg.get("num_heads", 8),
        head_dim=cfg.get("head_dim", 16),
        num_layers=cfg.get("num_layers", 5),
        num_legendre_polys=cfg.get("num_legendre_polys", 10),
        grid_shape=tuple(cfg.get("grid_shape", [32, 32])),
        window_sizes=tuple(cfg.get("window_sizes", [4, 4])),
        shift_sizes=tuple(cfg.get("shift_sizes", [2, 2])),
        top_k_ctot=cfg.get("top_k_ctot", 9),
    )
    log.info("Parameters: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    now_str = datetime.now().strftime("%Y%m%d")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    dataset_tag = f"{full_ds.config.dataset_type}_{'_'.join(obs_vars)}"
    run_name = f"ootg-baseline_{dataset_tag}_{now_str}_job{job_id}"
    group = cfg.logging.get("group", "ootg_baseline")

    checkpoint_dir = f"checkpoints/{group}/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config and normalization stats so evaluate_test_set.py can load this checkpoint.
    # Guard to rank 0: torch.save is not atomic, and all ranks have identical content.
    if int(os.environ.get("SLURM_PROCID", 0)) == 0:
        OmegaConf.save(cfg, os.path.join(checkpoint_dir, "config.yaml"))
    _coord_fn_name = (
        "lv95_to_latlon"
        if full_ds.config.dataset_type == "cosmo_peakweather"
        else "lonlat_swap"
    )
    if int(os.environ.get("SLURM_PROCID", 0)) == 0:
        torch.save(
            {
                "obs_mean": obs_mean,
                "obs_std": obs_std,
                "nwp_mean": nwp_mean,
                "nwp_std": nwp_std,
                "xc_grid_base": gen_train.xc_grid_latlon,
                "obs_vars": obs_vars,
                "dist_names": list(gen_train.dist_names),
                "n_nwp": dim_gridded_y,
                "nwp_stride": gen_train.nwp_stride,
                "coord_fn_name": _coord_fn_name,
                "model_kwargs": {
                    "dim_y": dim_y,
                    "dim_gridded_y": dim_gridded_y,
                    "obs_variable_to_distribution": v2d,
                    "embed_dim": cfg.get("embed_dim", 128),
                    "x_embed_dim": cfg.get("x_embed_dim", 128),
                    "num_heads": cfg.get("num_heads", 8),
                    "head_dim": cfg.get("head_dim", 16),
                    "num_layers": cfg.get("num_layers", 5),
                    "num_legendre_polys": cfg.get("num_legendre_polys", 10),
                    "grid_shape": tuple(cfg.get("grid_shape", [32, 32])),
                    "window_sizes": tuple(cfg.get("window_sizes", [4, 4])),
                    "shift_sizes": tuple(cfg.get("shift_sizes", [2, 2])),
                    "top_k_ctot": cfg.get("top_k_ctot", 9),
                },
            },
            os.path.join(checkpoint_dir, "ootg_stats.pt"),
        )
        log.info("Saved config.yaml and ootg_stats.pt to %s", checkpoint_dir)

    enable_wandb = cfg.logging.get("enable_wandb", True)
    wandb_logger = (
        WandbLogger(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            group=group,
            name=run_name,
        )
        if enable_wandb
        else False
    )

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

    n_devices = cfg.get("devices", 1)
    trainer = pl.Trainer(
        max_epochs=cfg.get("epochs", 300),
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.get("gradient_clip_val", 0.5),
        accelerator="auto",
        devices=n_devices,
        strategy="ddp" if n_devices > 1 else "auto",
        check_val_every_n_epoch=10,
        log_every_n_steps=10,
        limit_train_batches=gen_train.num_batches,
        limit_val_batches=gen_val.num_batches,
        enable_progress_bar=True,
    )
    predictor = OOTGPredictor(
        model=model,
        obs_vars=obs_vars,
        obs_mean=obs_mean,
        obs_std=obs_std,
        nwp_mean=nwp_mean,
        nwp_std=nwp_std,
        xc_grid_base=gen_train.xc_grid_latlon,
        coord_to_latlon=gen_train.coord_to_latlon,
        dist_names=list(gen_train.dist_names),
        n_nwp=dim_gridded_y,
        nwp_stride=gen_train.nwp_stride,
        output_transform=output_transform,
        lr=cfg.get("lr", 5e-4),
        n_samples=cfg.get("crps_mc_samples", 50),
    )
    trainer.fit(
        predictor,
        DataLoader(gen_train, batch_size=None, num_workers=0, pin_memory=True),
        DataLoader(gen_val, batch_size=None, num_workers=0, pin_memory=True),
    )


if __name__ == "__main__":
    main()
