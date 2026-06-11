import logging
from typing import Any

from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from oat.data_module.cosmo_peakweather_dataset import CosmoPeakWeatherDataset
from oat.data_module.data_split import create_splits
from oat.data_module.transform import NestedTransform, Transform


def _instantiate_dataset(cfg: DictConfig) -> Dataset[Any]:
    dataset_type = cfg.dataset.get("dataset_type", "cosmo_peakweather")
    if dataset_type == "era5_meteostat":
        from oat.data_module.era5_meteostat_dataset import ERA5MeteostatDataset

        return ERA5MeteostatDataset(cfg.dataset)
    if dataset_type == "cosmo_peakweather":
        return CosmoPeakWeatherDataset(cfg.dataset)
    raise ValueError(
        f"Unknown dataset_type: {dataset_type!r}. "
        "Supported: 'cosmo_peakweather', 'era5_meteostat'."
    )


def setup_data_and_transform(cfg: DictConfig, logger: logging.Logger) -> tuple[
    CosmoPeakWeatherDataset,
    int,
    DataLoader,
    DataLoader,
    DataLoader,
    NestedTransform,
    Transform,
]:
    logger.info("Setting up dataset and dataloaders...")
    dataset = _instantiate_dataset(cfg)
    logger.info(f"Dataset size: {len(dataset)} samples")
    splits = create_splits(dataset, **cfg.split, logger=logger)
    train_dataset, val_dataset, test_dataset = splits["datasets"]
    train_station_indices, val_station_indices, test_station_indices = splits["station_indices"]

    logger.info(f"Train size: {len(train_dataset)} samples")
    logger.info(f"Validation size: {len(val_dataset)} samples")
    logger.info(f"Test size: {len(test_dataset)} samples")

    if all(
        x is None
        for x in (
            cfg.split.train_station_indices,
            cfg.split.val_station_indices,
            cfg.split.test_station_indices,
        )
    ):
        logger.info("Adding station indices to the configuration.")
        cfg.split.train_station_indices = train_station_indices
        cfg.split.val_station_indices = val_station_indices
        cfg.split.test_station_indices = test_station_indices
    else:
        logger.info("Station indices already configured.")

    old_split_fraction = train_dataset.split_fraction
    train_dataset.split_fraction = 1.0
    logger.info("Temporarily set train datasets' split fraction to 1.0.")
    input_transform, output_transform = train_dataset.get_transform(
        num_data_points=cfg.dataset.data_points_for_stats,
        dataloader=DataLoader(train_dataset, shuffle=True, **cfg.dataloader.train),
        type=cfg.model.get("transform_type", "real"),
    )
    train_dataset.split_fraction = old_split_fraction
    logger.info(f"Set train datasets' split fraction back to {old_split_fraction}.")

    train_loader = DataLoader(train_dataset, shuffle=True, **cfg.dataloader.train)
    val_loader = DataLoader(val_dataset, shuffle=False, **cfg.dataloader.validation)
    test_loader = DataLoader(test_dataset, shuffle=False, **cfg.dataloader.test)

    return (
        dataset,
        len(train_dataset),
        train_loader,
        val_loader,
        test_loader,
        input_transform,
        output_transform,
    )
