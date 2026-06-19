import os
from datetime import datetime
import warnings

from oat.model_module import model
from oat.model_module.model import get_model_cls
from oat.data_module.setup import setup_data_and_transform
from oat.predictor_module.oat_predictor import setup_oat_cfg, setup_oat_predictor
from oat.predictor_module.gp_predictor import setup_gp_predictor
from oat.data_module.transform import Transform
from oat.predictor_module import OATPredictor
from oat.predictor_module.schedulers import estimate_total_steps
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import logging
from pathlib import Path
import pickle
from typing import Dict, Tuple, List, Any
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.profilers import Profiler
from torch.utils.data import DataLoader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_profiler(cfg: DictConfig) -> Profiler | None:
    if cfg.profiler.type == "simple":
        dirpath = os.path.join("profiler_outputs", cfg.logging.group)
        
        now_str = datetime.now().strftime("%Y%m%d")
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    
        filename = f"{cfg.logging.run_name_prefix}_{now_str}_job{slurm_job_id}"
        
        return SimpleProfiler(dirpath=dirpath, filename=filename)

    if cfg.profiler.type == "none" or cfg.profiler.type is None:
        return None

    raise ValueError(f"Profiler type {cfg.profiler.type} unknown.")
        
def setup_logger_and_callbacks(cfg: DictConfig) -> Tuple[WandbLogger, List]:
    base_name = cfg.logging.run_name_prefix
    now_str = datetime.now().strftime("%Y%m%d")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_name = f"{base_name}_{now_str}_job{slurm_job_id}"
    checkpoint_dir = os.path.join("checkpoints", cfg.logging.group, run_name)
    wandb_logger = WandbLogger(name=run_name,
                         project=cfg.logging.project,
                         group=cfg.logging.group,
                         entity=cfg.logging.entity,
                         log_model=False)

    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
        dirpath=checkpoint_dir,
        filename="best-model-{epoch:02d}-{val_loss:.4f}"
    )
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=None,
        save_top_k=0,
        save_last=True,
        every_n_epochs=1, # NOTE is epochs become long, consider saving every n_steps instead.
        enable_version_counter=False,
    )
    
    # NOTE maybe use cfg for patience
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=40,
        mode="min",
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    OmegaConf.save(cfg, os.path.join(checkpoint_dir, "config.yaml"))
    return wandb_logger, [checkpoint_callback, last_checkpoint_callback, early_stop_callback, lr_monitor]


def save_fixtures_for_tests(input_transform: Transform, output_transform: Transform, train_loader: DataLoader, cfg: DictConfig) -> None:
    assert cfg.dataloader.train.batch_size == 1, "Please use batch size = 1 when saving fixtures."

    logger.info("Saving examples of transforms as fixtures ...")
    ROOT = Path(__file__).resolve().parents[1]

    out_file = ROOT / "test" / "oat_test" / "input_transform_fixture.pickle"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(input_transform, f)
    logger.info(f"Saved input_transform object to {out_file}")

    out_file = ROOT / "test" / "oat_test" / "output_transform_fixture.pickle"
    with open(out_file, "wb") as f:
        pickle.dump(output_transform, f)
    logger.info(f"Saved output_transform object to {out_file}")

    batch = next(iter(train_loader))
    out_file = ROOT / "test" / "oat_test" / "example_batch_fixture.pickle"
    with open(out_file, "wb") as f:
        pickle.dump(batch, f)
    logger.info(f"Saved batch fixture object to {out_file}")



@hydra.main(config_path="../configs/", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # set global seed
    OmegaConf.resolve(cfg) # for referenced fields
    torch.manual_seed(cfg.seed)
    model_type = HydraConfig.get().runtime.choices.get("model", "oat")

    # Get model class
    model_cls = get_model_cls(cfg)
    # Check that only "informed" models have a train_split_fraction > 0.0
    # and force it to 0.0 otherwise.
    if not model_cls.requires_observation and cfg.split.train_split_fraction > 0.0:
        warnings.warn(f"Observation fraction = {cfg.split.train_split_fraction} is incompatible with \
            an uninformed model ({model_cls}). It will be forcibly overridden to 0.0")
        cfg.split.train_split_fraction = 0.0
        
    dataset, len_train_dataset, train_loader, val_loader, test_loader, input_transform, output_transform = setup_data_and_transform(cfg, logger)
    
    if cfg.get("save_fixtures", False):
        save_fixtures_for_tests(input_transform=input_transform,
                                output_transform=output_transform,
                                train_loader=train_loader,
                                cfg=cfg)
        return

    field_extent = {k: v.tolist() for k, v in dataset.nwp_extent.items()}
    wandb_logger, callbacks = setup_logger_and_callbacks(cfg)
    
    # print config
    logger.info(f"Config :\n {OmegaConf.to_yaml(cfg)}")

    if model_type == "baseline":
        from oat.model_module.baseline import NWPInterpolatorConfig
        from oat.predictor_module.baseline_predictor import create_baseline_predictor

        baseline_config = NWPInterpolatorConfig(
            field_extent=field_extent,
            **cfg.model
        )
        pre_transform = dataset.get_baseline_pre_transform()
        predictor = create_baseline_predictor(
            config=baseline_config,
            obs_vars=list(dataset.config.obs_vars),
            pre_transform=pre_transform,
        )

        trainer = Trainer(**cfg.trainer, logger=wandb_logger)
        trainer.validate(predictor, val_loader)
    elif model_type == "gp_baseline":
        predictor = setup_gp_predictor(
            cfg=cfg,
            field_extent=field_extent,
            obs_vars=list(dataset.config.obs_vars),
            transforms=[input_transform, output_transform],
        )

        profiler = setup_profiler(cfg=cfg)

        trainer = Trainer(
            **cfg.trainer,
            callbacks=callbacks,
            logger=wandb_logger,
            profiler=profiler,
        )
        trainer.fit(
            model=predictor,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=cfg.get("resume_from_checkpoint", None),
        )

        if profiler is not None and trainer.is_global_zero:
            print(profiler.summary())
    else:
        oat_model_cfg = setup_oat_cfg(cfg, field_extent, logger)

        var_names = dataset.config.obs_vars
        
        # Estimate the total number of training steps. Can be used by the OAT loss scheduler if its total_steps is set to None.
        estimated_total_steps = estimate_total_steps(
            dataset_size=len_train_dataset,
            batch_size=cfg.dataloader.train.batch_size,
            max_epochs=cfg.trainer.max_epochs,
            max_steps=OmegaConf.select(cfg.trainer, "max_steps", default=None),
            accumulate_grad_batches=OmegaConf.select(cfg.trainer, "accumulate_grad_batches", default=1),
            num_nodes=OmegaConf.select(cfg.trainer, "num_nodes", default=1),
            num_gpus_per_node=OmegaConf.select(cfg.trainer, "devices", default=4),
            limit_train_batches=OmegaConf.select(cfg.trainer, "limit_train_batches", default=1.0),
            drop_last=OmegaConf.select(cfg.dataloader.train, "drop_last", default=False)
        )
        logger.info(f"Manual estimated total optimizer steps: {estimated_total_steps}")

        predictor = setup_oat_predictor(model_cfg=oat_model_cfg,
                                        estimated_total_step=estimated_total_steps,
                                        var_names=var_names,
                                        transforms=[input_transform, output_transform],
                                        cfg=cfg)


        # NOTE check if compiling model helps
        # predictor = torch.compile(predictor)

        profiler = setup_profiler(cfg=cfg)

        trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=wandb_logger, profiler=profiler)
        trainer.fit(
            model=predictor,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=cfg.get("resume_from_checkpoint", None),
        )

        if profiler is not None and trainer.is_global_zero:
            print(profiler.summary())

if __name__ == "__main__":
    # Usage:
    # python -m scripts.train_model                              (OAT training)
    # python -m scripts.train_model model=baseline logging=baseline  (baseline evaluation)
    # python -m scripts.train_model +saves_fixtures=true ++dataloader.batch_size=1
    main()
