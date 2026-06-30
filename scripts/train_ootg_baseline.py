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
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from oat.baselines.ootg_generator import compute_norm_stats, make_ootg_generator
from oat.baselines.ootg_likelihood import JointMixedDistribution
from oat.baselines.ootg_model import build_ootg_model
from oat.data_module.data_split import create_splits
from oat.data_module.setup import _instantiate_dataset

log = logging.getLogger(__name__)


class OOTGLitModule(pl.LightningModule):
    def __init__(self, model, lr: float = 5e-4) -> None:
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, batch) -> JointMixedDistribution:
        return self.model(batch.xc, batch.yc, batch.xc_grid, batch.yc_grid, batch.xt)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        dist = self(batch)
        loss = -dist.log_prob(batch.yt).sum(-1).mean()
        self.log("train/nll", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        dist = self(batch)
        nll = -dist.log_prob(batch.yt).sum(-1).mean()
        rmse = (dist.mean - batch.yt).pow(2).mean().sqrt()
        self.log("val/nll", nll, on_epoch=True, prog_bar=True)
        self.log("val/rmse", rmse, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    log.info("Loading dataset...")
    full_ds = _instantiate_dataset(cfg)
    splits = create_splits(full_ds, **cfg.split, logger=log)
    train_wrap, val_wrap, _ = splits["datasets"]

    obs_vars = list(full_ds.config.obs_vars)
    nwp_vars = list(full_ds.config.nwp_vars)
    v2d = dict(full_ds.config.obs_variable_to_distribution)
    dim_y, dim_gridded_y = len(obs_vars), len(nwp_vars)
    log.info("obs_vars=%s  v2d=%s  nwp_vars=%s", obs_vars, v2d, nwp_vars)

    log.info("Computing normalization stats (n=%d)...", cfg.get("norm_samples", 500))
    obs_mean, obs_std, nwp_mean, nwp_std = compute_norm_stats(
        full_ds, n_samples=cfg.get("norm_samples", 500),
        time_indices=train_wrap._time_indices,
    )
    log.info("obs_mean=%s  obs_std=%s", obs_mean.tolist(), obs_std.tolist())

    gen_train = make_ootg_generator(
        full_ds, obs_mean=obs_mean, obs_std=obs_std, nwp_mean=nwp_mean, nwp_std=nwp_std,
        nwp_stride=cfg.get("nwp_stride", None), time_indices=train_wrap._time_indices,
        pc_context=cfg.get("pc_context", 0.8),
        samples_per_epoch=cfg.get("samples_per_epoch", 10000),
    )
    gen_val = make_ootg_generator(
        full_ds, obs_mean=obs_mean, obs_std=obs_std, nwp_mean=nwp_mean, nwp_std=nwp_std,
        nwp_stride=cfg.get("nwp_stride", None), time_indices=val_wrap._time_indices,
        samples_per_epoch=cfg.get("val_samples_per_epoch", 1000),
    )

    model = build_ootg_model(
        dim_y=dim_y, dim_gridded_y=dim_gridded_y,
        obs_variable_to_distribution=v2d,
        embed_dim=cfg.get("embed_dim", 128), x_embed_dim=cfg.get("x_embed_dim", 128),
        num_heads=cfg.get("num_heads", 8), head_dim=cfg.get("head_dim", 16),
        num_layers=cfg.get("num_layers", 5), num_legendre_polys=cfg.get("num_legendre_polys", 10),
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

    enable_wandb = cfg.logging.get("enable_wandb", True)
    wandb_logger = (
        WandbLogger(project=cfg.logging.project, entity=cfg.logging.entity, group=group, name=run_name)
        if enable_wandb else False
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=f"checkpoints/{group}/{run_name}",
            filename="epoch={epoch}-val_nll={val/nll:.4f}",
            monitor="val/nll", mode="min", save_top_k=2,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.get("epochs", 300), logger=wandb_logger, callbacks=callbacks,
        gradient_clip_val=cfg.get("gradient_clip_val", 0.5),
        accelerator="auto", devices=1, check_val_every_n_epoch=10, log_every_n_steps=10,
        limit_train_batches=gen_train.num_batches, limit_val_batches=gen_val.num_batches,
        enable_progress_bar=True,
    )
    trainer.fit(
        OOTGLitModule(model, lr=cfg.get("lr", 5e-4)),
        DataLoader(gen_train, batch_size=None, num_workers=0, pin_memory=True),
        DataLoader(gen_val, batch_size=None, num_workers=0, pin_memory=True),
    )


if __name__ == "__main__":
    main()
