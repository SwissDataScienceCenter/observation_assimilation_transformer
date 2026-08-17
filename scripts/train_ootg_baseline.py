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
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from oat.baselines.ootg_generator import compute_norm_stats, make_ootg_generator
from oat.baselines.ootg_model import build_ootg_model
from oat.baselines.ootg_predictor import OOTGPredictor
from oat.data_module.legacy.data_split import create_splits
from oat.data_module.setup import _instantiate_dataset

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="ootg_baseline", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    # Seed so the random train/val station partition (create_splits uses the
    # global RNG) is identical across runs — required for a valid ablation
    # comparison where only the information regime should differ. Default 42
    # matches OAT (configs/config.yaml) so OOTG and OAT share the split.
    torch.manual_seed(cfg.get("seed", 42))
    log.info("Loading dataset...")
    full_ds = _instantiate_dataset(cfg)
    splits = create_splits(full_ds, **cfg.split, logger=log)
    train_wrap, val_wrap, _ = splits["datasets"]
    train_station_indices, val_station_indices, test_station_indices = splits[
        "station_indices"
    ]
    train_station_indices = np.array(train_station_indices)
    val_station_indices = np.array(val_station_indices)

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

    post_transform, output_transform = full_ds.get_transform(
        num_data_points=cfg.get("norm_samples", 500),
        data_indices=train_wrap._time_indices,
        type=cfg.get("transform_type", "keep_support"),
    )

    use_topo = cfg.get("use_topo", False)
    topo_transform = None
    topo_in_channels = None
    topo_initial_hw = None
    if use_topo:
        topo_transform = post_transform.transform_dict[
            "observation_data"
        ].transform_dict["topo"]
        # Derive patch geometry from an actual sample (dataset-agnostic).
        sample_topo = full_ds[int(train_wrap._time_indices[0])]["observation_data"][
            "topo"
        ]
        topo_in_channels = sample_topo.shape[-3]
        topo_initial_hw = (sample_topo.shape[-2], sample_topo.shape[-1])
        log.info(
            "use_topo=True  C_topo=%d  patch_hw=%s",
            topo_in_channels,
            topo_initial_hw,
        )

    # Ablations for H2 (NWP-grid) vs H3 (observation-assimilation) diagnosis.
    # Applied to BOTH train and val generators so the model is trained and
    # scored under the same information regime.
    ablate_obs = cfg.get("ablate_obs", False)
    ablate_grid = cfg.get("ablate_grid", False)
    if ablate_obs and ablate_grid:
        raise ValueError("ablate_obs and ablate_grid together leave no signal.")
    if ablate_obs or ablate_grid:
        log.info("ABLATION active: ablate_obs=%s ablate_grid=%s", ablate_obs, ablate_grid)

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
        eligible_station_indices=train_station_indices,
        ablate_obs=ablate_obs,
        ablate_grid=ablate_grid,
        topo_transform=topo_transform,
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
        context_station_indices=train_station_indices,
        target_station_indices=val_station_indices,
        ablate_obs=ablate_obs,
        ablate_grid=ablate_grid,
        topo_transform=topo_transform,
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
        use_topo=use_topo,
        topo_in_channels=topo_in_channels,
        topo_out_channels=cfg.get("topo_out_channels", 32),
        topo_n_layers=cfg.get("topo_n_layers", 3),
        topo_kernel_size=cfg.get("topo_kernel_size", 3),
        topo_initial_hw=topo_initial_hw,
    )
    log.info("Parameters: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    now_str = datetime.now().strftime("%Y%m%d")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    ablation_tag = (
        "_nwponly" if ablate_obs else "_obsonly" if ablate_grid else ""
    )
    dataset_tag = f"{full_ds.config.dataset_type}_{'_'.join(obs_vars)}"
    default_run_name = f"ootg-baseline{ablation_tag}_{dataset_tag}_{now_str}_job{job_id}"
    run_name = cfg.get("run_name") or default_run_name
    group = cfg.logging.get("group", "ootg_baseline")

    checkpoint_dir = f"checkpoints/{group}/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config and normalization stats so evaluate_test_set.py can load this checkpoint.
    # Guard to rank 0: torch.save is not atomic, and all ranks have identical content.
    if int(os.environ.get("SLURM_PROCID", 0)) == 0:
        # Persist the resolved station split (regenerated from the global seed
        # otherwise) and a `model.transform_type` key so evaluate_test_set.py can
        # drive setup_data_and_transform from this config alone. `dataloader` is
        # provided via the config defaults; it is unused during training.
        with open_dict(cfg):
            cfg.split.train_station_indices = [int(i) for i in train_station_indices]
            cfg.split.val_station_indices = [int(i) for i in val_station_indices]
            cfg.split.test_station_indices = [int(i) for i in test_station_indices]
            cfg.model = {"transform_type": cfg.get("transform_type", "keep_support")}
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
                "output_transform": (
                    output_transform.to_dict() if output_transform is not None else None
                ),
                "topo_transform": (
                    topo_transform.to_dict() if topo_transform is not None else None
                ),
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
                    "use_topo": use_topo,
                    "topo_in_channels": topo_in_channels,
                    "topo_out_channels": cfg.get("topo_out_channels", 32),
                    "topo_n_layers": cfg.get("topo_n_layers", 3),
                    "topo_kernel_size": cfg.get("topo_kernel_size", 3),
                    "topo_initial_hw": topo_initial_hw,
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
        check_val_every_n_epoch=cfg.get("check_val_every_n_epoch", 10),
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
        use_topo=use_topo,
        topo_transform=topo_transform,
    )
    trainer.fit(
        predictor,
        DataLoader(gen_train, batch_size=None, num_workers=0, pin_memory=True),
        DataLoader(gen_val, batch_size=None, num_workers=0, pin_memory=True),
    )


if __name__ == "__main__":
    main()
