import os
from datetime import datetime

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
from oat.model_module import baseline

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
    val_loader = DataLoader(val_dataset, shuffle=True, **cfg.dataloader)
    test_loader = DataLoader(test_dataset, shuffle=False, **cfg.dataloader)
    logger.info(f"Train loader batches: {len(train_loader)}")
    logger.info(f"Validation loader batches: {len(val_loader)}")
    logger.info(f"Test loader batches: {len(test_loader)}")
    
    # TODO: Move the number somewhere to config
    input_transform, output_transform = train_dataset.get_transform(num_data_points=64,#cfg.dataset.data_points_for_stats,
                                                                    dataloader=train_loader)
    logger.info("Transforms set up successfully.")
    return dataset, train_loader, val_loader, test_loader, input_transform, output_transform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs/", config_name="baseline_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # set global seed
    OmegaConf.resolve(cfg) # for referenced fields
    torch.manual_seed(cfg.seed)
    dataset, train_loader, val_loader, test_loader, input_transform, output_transform = setup_data_and_transform(cfg, logger)
    field_extent = {k: v.tolist() for k, v in dataset.nwp_extent.items()}
    model_config = baseline.InterpolaterBaselineConfig(
        field_extent=field_extent,
        **cfg.model
    )
    baseline_model = dataset.get_baseline_predictor(model_config)
    base_name = cfg.logging.run_name_prefix
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_name = f"{base_name}_{now_str}_job{slurm_job_id}"
    wandb_logger = WandbLogger(name=run_name,
                         project=cfg.logging.project,
                         group=cfg.logging.group,
                         entity=cfg.logging.entity,
                         log_model=False)
    
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    
    trainer = Trainer(**cfg.trainer, logger=wandb_logger)
    
    trainer.validate(baseline_model, val_loader)
    
    
if __name__ == "__main__":
    main()