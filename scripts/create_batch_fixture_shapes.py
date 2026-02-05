from pathlib import Path

import yaml
from oat.data_module.cosmo_peakweather_dataset import CosmoPeakWeatherConfig, CosmoPeakWeatherDataset
from oat.data_module.data_split import create_splits
import torch
from torch.utils.data import DataLoader
import pandas as pd
from omegaconf import OmegaConf
from src.oat.data_module import factory

def main():
    config_path = "/scratch/mch/cdonner/observation_assimilation_transformer/configs/data/cosmo_peakweather_data.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    config = CosmoPeakWeatherConfig(**config_dict)

    dataset = CosmoPeakWeatherDataset(config)
    train_dataset, val_dataset, test_dataset = create_splits(dataset)
    datasets = [train_dataset, val_dataset, test_dataset]
    labels = ["train_dataset", "val_dataset", "test_dataset"]
    

    for ds, label in zip(datasets, labels):
    
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        batch = next(iter(loader))


        df = create_batch_fixture_table(batch=batch)

        ROOT = Path(__file__).resolve().parents[1]

        out_file = ROOT / "test" / "oat_test" / f"batch_shapes_{label}.pickle"

        out_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_pickle(out_file)

        print(f"Saved batch shape table to {out_file}")

        df = pd.read_pickle(out_file)

        print("With content:\n", df)


def create_batch_fixture_table(batch: dict[str, dict[str, torch.Tensor]]) -> pd.DataFrame:

    rows = []
    for k, v in batch.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                rows.append({
                    "group": k,
                    "key": kk,
                    "shape": tuple(vv.shape),
                    "dtype": vv.dtype if hasattr(vv, "dtype") else None,
                })
        else:
            rows.append({
                    "group": k,
                    "key": k,
                    "shape": tuple(v.shape),
                    "dtype": v.dtype if hasattr(v, "dtype") else None,
                })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    """
    Generates a table with the key-value pairs, shapes and data type for an example batch.
    This table is then used by the tests to generate dummy data.
    Re-run this every time the data loader is updated and the shapes might have changed.
    Usage:
        python -m scripts.create_batch_fixture_shapes
    """
    main()