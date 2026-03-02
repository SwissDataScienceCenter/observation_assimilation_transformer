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

    # NOTE: to compute the statistics, only "observation_data" is used.
    # for this reason, we temporarily ignore the split_fraction and set it to 1.0
    # so we use 100% of the stations as "conditioning" stations to compute 
    # the statistics. We then set it back.
    old_split_fraction = train_dataset.split_fraction
    train_dataset.split_fraction = 1.0
    logger.info(f"Temporarily set train datasets' split fraction to 1.0.")
    # NOTE should we  use a DataLoader in get_transform or not? (its optional)
    # This changed in the merge of the uninformed baseline. TODO check it.
    input_transform, output_transform = train_dataset.get_transform(
        num_data_points=cfg.dataset.data_points_for_stats, 
        dataloader=DataLoader(train_dataset, shuffle=True, **cfg.dataloader)
    )
    train_dataset.split_fraction = old_split_fraction
    logger.info("Transforms set up successfully.")
    logger.info(f"Set train datasets' split fraction back to {old_split_fraction}.")

    train_loader = DataLoader(train_dataset, shuffle=True, **cfg.dataloader)
    val_loader = DataLoader(val_dataset, shuffle=True, **cfg.dataloader)
    test_loader = DataLoader(test_dataset, shuffle=False, **cfg.dataloader)
    logger.info(f"Train loader batches: {len(train_loader)}")
    logger.info(f"Validation loader batches: {len(val_loader)}")
    logger.info(f"Test loader batches: {len(test_loader)}")

    return dataset, train_loader, val_loader, test_loader, input_transform, output_transform
