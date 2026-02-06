import pytest
from pathlib import Path
import torch
import pickle

from oat.data_module.transform import Transform

@pytest.fixture
def batch_fixture() -> dict[str, dict[str, torch.Tensor]]:
    """Provides a dummy batch based on the saved batch_shapes.pickle table."""
    # return generate_batch_from_table()
    
    batch_fixture_file = Path(__file__).resolve().parent / "example_batch_fixture.pickle"
    with open(batch_fixture_file, "rb") as f:
        batch = pickle.load(f)
    return batch 

@pytest.fixture
def transform_fixture() -> tuple[Transform, Transform]:
    """Provides a dummy transform."""
    
    in_pickle_file = Path(__file__).resolve().parent / "input_transform_fixture.pickle"
    out_pickle_file = Path(__file__).resolve().parent / "output_transform_fixture.pickle"
    
    with open(in_pickle_file, "rb") as f:
        in_transform = pickle.load(f) 
        
    with open(out_pickle_file, "rb") as f:
        out_transform = pickle.load(f) 
        
    return in_transform, out_transform
