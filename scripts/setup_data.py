from oat.data_module.cosmo_peakweather_dataset import CosmoPeakWeatherDataset
from oat.data_module.data_split import create_splits
from omegaconf import DictConfig
import hydra
import torch
import logging

logging.basicConfig(level=logging.INFO)

@hydra.main(config_path="../configs/", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # set global seed
    torch.manual_seed(cfg.seed)
    logger = logging.getLogger(__name__)
    logger.info("Setting up dataset and dataloaders...")
    dataset = CosmoPeakWeatherDataset(cfg.dataset)
    logger.info(f"Dataset size: {len(dataset)} samples")
    train_dataset, val_dataset, test_dataset = create_splits(dataset, **cfg.split)
    logger.info(f"Train size: {len(train_dataset)} samples")
    logger.info(f"Validation size: {len(val_dataset)} samples")
    logger.info(f"Test size: {len(test_dataset)} samples")
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **cfg.dataloader)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **cfg.dataloader)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **cfg.dataloader)
    logger.info(f"Train loader batches: {len(train_loader)}")
    logger.info(f"Validation loader batches: {len(val_loader)}")
    logger.info(f"Test loader batches: {len(test_loader)}")
    
    # TODO: Move the number somewhere to config
    input_transform, output_transform = train_dataset.get_transform(num_data_points=cfg.dataset.data_points_for_stats,
                                                                    dataloader=train_loader)
    logger.info("Transforms set up successfully.")
    
    # Example: Visualize a sample from the training dataset
    sample = train_dataset[0]
    input_data = input_transform(sample)
    output_data = output_transform(sample['observation_data']['obs'])
    
    logger.info(f"Input Data Keys: {input_data.keys()}")
    logger.info(f"Output Data Keys: {output_data.shape}")
    
if __name__ == "__main__":
    main()