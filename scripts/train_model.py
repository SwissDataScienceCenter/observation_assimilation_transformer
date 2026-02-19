import os
from datetime import datetime

from oat.model_module import model
from oat.data_module.setup import setup_data_and_transform
from oat.data_module.transform import Transform
from oat.predictor_module import OATPredictor
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import logging
from pathlib import Path
import pickle
from typing import Dict, Tuple, List
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader


def setup_oat_cfg(cfg: DictConfig, field_extent: Dict, logger: logging.Logger) -> model.OATConfig:
    model_cfg = cfg.model
    data_cfg = cfg.dataset
    logger.info("Setting up model configs...")
    global_input_chans = len(data_cfg.nwp_vars) + len(data_cfg.topo_vars)
    global_encoder_cfg = model.GlobalEncoderConfig(
        **model_cfg.global_encoder,
        in_channels=global_input_chans,
        field_extent_max=field_extent['max'],
        field_extent_min=field_extent['min']
        )
    topography_encoder_cfg = model.TopographicEncoderConfig(
        **model_cfg.topographic_encoder,
        in_channels=len(data_cfg.topo_vars)
        )
    fusion_layer_cfg = model.FusionLayerConfig(
        **model_cfg.fusion_layer,
        out_channels=model_cfg.d_model)
    masking_cfg = model.MaskingConfig(**model_cfg.masking)
    transformer_cfg = model.TransformerConfig(
        d_geo_bias=len(data_cfg.geo_vars),
        d_model=model_cfg.d_model,
        **model_cfg.transformer)
    output_head_cfg = model.OutputHeadConfig(
        target_features=len(data_cfg.obs_vars),
        in_features=model_cfg.d_model,
        **model_cfg.output_head)
    oat_config = model.OATConfig(
        global_encoder=global_encoder_cfg,
        topographic_encoder=topography_encoder_cfg,
        fusion_layer=fusion_layer_cfg,
        masking=masking_cfg,
        transformer=transformer_cfg,
        output_head=output_head_cfg,
        num_samples=model_cfg.num_samples,
        pre_oat_loss_weight=model_cfg.pre_oat_loss_weight
    )
    logger.info("Initialized model cfg...")
    return oat_config

def setup_oat_predictor(model_cfg: model.OATConfig, 
                    var_names: str,
                    transforms: tuple[Transform], cfg: DictConfig) -> OATPredictor:

    from importlib import import_module

    # optimizer
    module_path, class_name = cfg.predictor.kwargs.optimizer_class.rsplit(".", 1)
    module = import_module(module_path)
    optim_cls = getattr(module, class_name)

    # scheduler
    module_path, class_name = cfg.predictor.kwargs.scheduler_class.rsplit(".", 1)
    module = import_module(module_path)
    sched_cls = getattr(module, class_name)


    predictor = OATPredictor(
        oat_model_cfg=model_cfg,
        input_transform=transforms[0],
        output_transform=transforms[1],
        output_var_names=var_names,
        lr=cfg.predictor.lr,
        optimizer_class=optim_cls,
        optimizer_params=cfg.predictor.kwargs.optimizer_params,
        scheduler_class=sched_cls,
        scheduler_params=cfg.predictor.kwargs.scheduler_params
    )

    return predictor


def setup_logger_and_callbacks(cfg: DictConfig) -> Tuple[WandbLogger, List]:
    base_name = cfg.logging.run_name_prefix
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_name = f"{base_name}_{now_str}_job{slurm_job_id}"
    checkpoint_dir = os.path.join("checkpoints", cfg.logging.group, run_name)
    wandb_logger = WandbLogger(name=run_name,
                         project=cfg.logging.project,
                         group=cfg.logging.group,
                         entity=cfg.logging.entity,
                         log_model=False)

    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    
            # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
        dirpath=checkpoint_dir,
        filename="best-model-{epoch:02d}-{val_loss:.4f}"
    )
    # NOTE maybe use cfg for patience
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    return wandb_logger, [checkpoint_callback, early_stop_callback, lr_monitor]


def save_fixtures_for_tests(input_transform: Transform, output_transform: Transform, train_loader: DataLoader, cfg: DictConfig) -> None:
    assert cfg.dataloader.batch_size == 1, "Please use batch size = 1 when saving fixtures."

    logger.info("Saving examples of transforms as fixtures ...")
    ROOT = Path(__file__).resolve().parents[1]
    out_file = ROOT / "test" / "oat_test" / f"input_transform_fixture.pickle"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_file, "wb") as f:
        pickle.dump(input_transform, f)
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
    logger.info(f"Saved input_transform object to {out_file}")
    
    out_file = ROOT / "test" / "oat_test" / f"output_transform_fixture.pickle"
    
    with open(out_file, "wb") as f:
        pickle.dump(output_transform, f)

    
    logger.info(f"Saved output_transform object to {out_file}")
    batch = next(iter(train_loader))
    out_file = ROOT / "test" / "oat_test" / f"example_batch_fixture.pickle"
    with open(out_file, "wb") as f:
            pickle.dump(batch, f)
            
    logger.info(f"Saved batch fixture object to {out_file}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs/", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # set global seed
    OmegaConf.resolve(cfg) # for referenced fields
    torch.manual_seed(cfg.seed)
    model_type = HydraConfig.get().runtime.choices.get("model", "oat")
    dataset, train_loader, val_loader, test_loader, input_transform, output_transform = setup_data_and_transform(cfg, logger)
    
    if cfg.get("save_fixtures", False):
        save_fixtures_for_tests(input_transform=input_transform,
                                output_transform=output_transform,
                                train_loader=train_loader,
                                cfg=cfg)
        return

    field_extent = {k: v.tolist() for k, v in dataset.nwp_extent.items()}
    wandb_logger, callbacks = setup_logger_and_callbacks(cfg)

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
    else:
        oat_model_cfg = setup_oat_cfg(cfg, field_extent, logger)
        logger.info("Model set up successfully.")

        var_names = dataset.config.obs_vars
        predictor = setup_oat_predictor(model_cfg=oat_model_cfg,
                                        var_names=var_names,
                                        transforms=[input_transform, output_transform],
                                        cfg=cfg)


        # NOTE check if compiling model helps
        # predictor = torch.compile(predictor)

        trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=wandb_logger)
        trainer.fit(model=predictor,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader)

if __name__ == "__main__":
    # Usage:
    # python -m scripts.train_model                              (OAT training)
    # python -m scripts.train_model model=baseline logging=baseline  (baseline evaluation)
    # python -m scripts.train_model +save_transforms_fixtures=true ++dataloader.batch_size=1
    main()
