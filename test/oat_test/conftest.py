import pytest
import pandas as pd
from pathlib import Path
import torch

# Path to the table containing the shapes and dtypes.
TABLE_PATH = Path(__file__).resolve().parent / "batch_shapes.pickle"

def generate_batch_from_table() -> dict[str, dict[str, torch.Tensor]]:
    """
    Reconstructs a batch dict example from batch_shapes.pickle.
    All tensors are filled with random values.
    """
    df = pd.read_pickle(TABLE_PATH)
    
    batch: dict[str, dict[str, torch.Tensor]] = {}
    for group_name in df['group'].unique():
        batch[group_name] = {}
        group_df = df[df['group'] == group_name]
        for _, row in group_df.iterrows():
            shape = tuple(row['shape'])

            if isinstance(row['dtype'], torch.dtype):
                dtype = row['dtype']
            elif isinstance(row['dtype'], str):
                dtype = getattr(torch, row['dtype']) if hasattr(torch, row['dtype']) else torch.float32
            else:
                dtype = torch.float32

            batch[group_name][row['key']] = torch.randn(shape, dtype=dtype)
    return batch

@pytest.fixture
def batch_fixture() -> dict[str, dict[str, torch.Tensor]]:
    """Provides a dummy batch based on the saved batch_shapes.pickle table."""
    return generate_batch_from_table()
