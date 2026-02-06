from oat.data_module.cosmo_peakweather_dataset import CosmoPeakWeatherDataset
from oat.data_module.data_split import create_splits
from oat.model_module import model
from oat.data_module.transform import NestedTransform, Transform
from oat.predictor_module import OATPredictor
from omegaconf import DictConfig
import hydra
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import pickle
from typing import Tuple, Dict
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import WandbLogger

def setup_data_and_transform(cfg: DictConfig, logger: logging.Logger) -> Tuple[CosmoPeakWeatherDataset, 
                                                            DataLoader, 
                                                            DataLoader, 
                                                            DataLoader, 
                                                            NestedTransform, 
                                                            Transform]:
    logger.info("Setting up dataset and dataloaders...")
    dataset = CosmoPeakWeatherDataset(cfg.dataset)
    logger.info(f"Dataset size: {len(dataset)} samples")
    train_dataset, val_dataset, test_dataset = create_splits(dataset, **cfg.split)
    logger.info(f"Train size: {len(train_dataset)} samples")
    logger.info(f"Validation size: {len(val_dataset)} samples")
    logger.info(f"Test size: {len(test_dataset)} samples")
    train_loader = DataLoader(train_dataset, shuffle=True, **cfg.dataloader)
    val_loader = DataLoader(val_dataset, shuffle=False, **cfg.dataloader)
    test_loader = DataLoader(test_dataset, shuffle=False, **cfg.dataloader)
    logger.info(f"Train loader batches: {len(train_loader)}")
    logger.info(f"Validation loader batches: {len(val_loader)}")
    logger.info(f"Test loader batches: {len(test_loader)}")
    
    # TODO: Move the number somewhere to config
    input_transform, output_transform = train_dataset.get_transform(num_data_points=64,#cfg.dataset.data_points_for_stats,
                                                                    dataloader=train_loader)
    logger.info("Transforms set up successfully.")
    return dataset, train_loader, val_loader, test_loader, input_transform, output_transform


def setup_model(cfg: DictConfig, field_extent: Dict, logger: logging.Logger) -> model.OAT:
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
        num_samples=model_cfg.num_samples
    )
    logger.info("Initializing model...")
    model_instance = model.OAT(oat_config)
    return model_instance

def setup_predictor(model: nn.Module, transforms: tuple[Transform], cfg: DictConfig):

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
        oat_model=model,
        input_transform=transforms[0],
        output_transform=transforms[1],
        lr=cfg.predictor.lr,
        optimizer_class=optim_cls,
        optimizer_params=cfg.predictor.kwargs.optimizer_params,
        scheduler_class=sched_cls,
        scheduler_params=cfg.predictor.kwargs.scheduler_params
    )
    
    return predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs/", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # set global seed
    OmegaConf.resolve(cfg) # for referenced fields
    torch.manual_seed(cfg.seed)
    dataset, train_loader, val_loader, test_loader, input_transform, output_transform = setup_data_and_transform(cfg, logger)
    if cfg.get("save_transforms_fixtures", False):
        assert cfg.dataloader.batch_size == 1, "Please use batch size = 1 when saving fixtures."

        logger.info("Saving examples of transforms as fixtures ...")
        ROOT = Path(__file__).resolve().parents[1]
        out_file = ROOT / "test" / "oat_test" / f"input_transform_fixture.pickle"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_file, "wb") as f:
            pickle.dump(input_transform, f)

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

    field_extent = {k: v.tolist() for k, v in dataset.nwp_extent.items()}
    oat_model = setup_model(cfg, field_extent, logger)
    logger.info("Model set up successfully.")
    
    
    
    predictor = setup_predictor(model=oat_model,
                                transforms=[input_transform, output_transform],
                                cfg=cfg)
    # Callbacks
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
        dirpath="checkpoints/",
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
    
    # Logger 
    wandb_logger = WandbLogger(name='first_run',
                         project='oat',
                         experiment='first_experiment',
                         log_model=False)
    
    trainer = Trainer(**cfg.trainer, callbacks=[checkpoint_callback,
                                                early_stop_callback,
                                                lr_monitor],
                                    logger=wandb_logger)

    trainer.fit(model=predictor,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)
                
if __name__ == "__main__":
    # Usage:
    # python -m scripts.setup_data +save_transforms_fixtures=true ++dataset.data_points_for_stats=10
    # python -m scripts.setup_data +save_transforms_fixtures=true ++dataset.data_points_for_stats=10 ++dataloader.batch_size=1
    main()