from pathlib import Path

import torch
import pandas as pd
from omegaconf import OmegaConf
from src.oat.data_module import factory

def main():
    data_config = OmegaConf.load('/scratch/mch/cdonner/observation_assimilation_transformer/configs/data/wind_data.yaml')
    train_dataloader, val_dataloader, test_dataloader = factory.create_dataset(data_config)
    batch = next(iter(train_dataloader))

    df = create_batch_fixture_table(batch=batch)

    ROOT = Path(__file__).resolve().parents[1]

    out_file = ROOT / "test" / "batch_shapes.pickle"

    out_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_pickle(out_file)

    print(f"Saved batch shape table to {out_file}")

    df = pd.read_pickle(out_file)

    print("With content:\n", df)


def create_batch_fixture_table(batch: dict[str, dict[str, torch.Tensor]]) -> pd.DataFrame:

    rows = []
    for k, v in batch.items():
        for kk, vv in v.items():
            rows.append({
                "group": k,
                "key": kk,
                "shape": tuple(vv.shape),
                "dtype": vv.dtype if hasattr(vv, "dtype") else None,
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