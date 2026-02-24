import logging
from typing import Tuple

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from oat.data_module.cosmo_peakweather_dataset import CosmoPeakWeatherDataset
from oat.data_module.data_split import create_splits
from oat.data_module.transform import NestedTransform, Transform


def setup_data_and_transform(
    cfg: DictConfig, logger: logging.Logger
) -> Tuple[
    CosmoPeakWeatherDataset,
    DataLoader,
    DataLoader,
    DataLoader,
    NestedTransform,
    Transform,
]:
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
    input_transform, output_transform = train_dataset.get_transform(num_data_points=64)
    logger.info("Transforms set up successfully.")
    return dataset, train_loader, val_loader, test_loader, input_transform, output_transform
