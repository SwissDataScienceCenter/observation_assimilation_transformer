import pytest
import pandas as pd
from pathlib import Path
import torch

# Path to the table containing the shapes and dtypes.
TABLE_PATH = Path(__file__).resolve().parent / "batch_shapes_test_dataset.pickle"

def generate_batch_from_table() -> dict[str, dict[str, torch.Tensor]]:
    """
    Reconstructs a batch dict example from batch_shapes.pickle.
    All float tensors are filled with random values.
    Non-float tensors are filled with zeros.
    """
    try:
        df = pd.read_pickle(TABLE_PATH)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not open batch shapes table at '{TABLE_PATH}'.\n"
            "Make sure the file exists and has been generated "
            "by running: python -m scripts.create_batch_fixture_shapes"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error while loading batch shapes table at '{TABLE_PATH}'.\n"
        ) from e

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

            if getattr(dtype, "is_floating_point", False):
                tensor = torch.randn(shape, dtype=dtype)
            elif row['key'] == 'index':
                tensor = torch.randperm(shape[0])
            else:
                # Fallback for non-floating dtypes: use zeros to avoid torch.randn errors.
                tensor = torch.zeros(shape, dtype=dtype)

            if group_name != row['key']:
                batch[group_name][row['key']] = tensor
            else:
                batch[group_name] = tensor
    return batch

@pytest.fixture
def batch_fixture() -> dict[str, dict[str, torch.Tensor]]:
    """Provides a dummy batch based on the saved batch_shapes.pickle table."""
    return generate_batch_from_table()
