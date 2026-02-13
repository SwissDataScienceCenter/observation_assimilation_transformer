from typing import Type, Tuple
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig

from oat.predictor_module import OATPredictor

def load_checkpoint(predictor_cls: Type[pl.LightningModule], checkpoint_file_path: str)-> Tuple[OATPredictor, DictConfig]:
    """
    Load and return the Predictor and its config from a checkpoint file.
    
    predictor_cls is the class of the Predictor to load. 
    checkpoint_file_path is the path to the checkpoint file, not the folder.
    It is expected that the parent folder of this checkpoint contains, at the same 
    level as the checkpoint, a yaml file with the config used to train the model.
    """
    path = Path(checkpoint_file_path)
    checkpoint_folder = path.parent
    config_path = checkpoint_folder / "config.yaml"
    
    cfg = OmegaConf.load(config_path)
    
    predictor = predictor_cls.load_from_checkpoint(path)
    
    return predictor, cfg