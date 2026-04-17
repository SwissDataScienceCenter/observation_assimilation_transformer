import importlib.util
from pathlib import Path

from oat.predictor_module.schedulers import estimate_total_steps

import pytest

_SCHEDULERS_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "oat"
    / "predictor_module"
    / "schedulers.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "oat.predictor_module.schedulers", _SCHEDULERS_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_SCHEDULERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCHEDULERS)

LinearWeightScheduler = _SCHEDULERS.LinearWeightScheduler
ExponentialWeightScheduler = _SCHEDULERS.ExponentialWeightScheduler


def test_linear_scheduler_warmup_and_clamp() -> None:
    scheduler = LinearWeightScheduler(
        start_weight=1.0, flat_warmup_steps=2, end_weight=0.0, total_steps=5
    )

    weights = [scheduler.step() for _ in range(8)]

    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(1.0)
    assert weights[2] == pytest.approx(1.0)
    assert weights[3] == pytest.approx(2.0 / 3.0)
    assert weights[4] == pytest.approx(1.0 / 3.0)
    assert weights[5] == pytest.approx(0.0)
    assert weights[6] == pytest.approx(0.0)
    assert weights[7] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("start_weight", "end_weight"),
    [(0.0, 0.1), (0.1, 0.0), (-0.1, 0.1), (0.1, -0.1)],
)
def test_exponential_scheduler_rejects_non_positive_weights(
    start_weight: float, end_weight: float
) -> None:
    with pytest.raises(ValueError):
        ExponentialWeightScheduler(
            start_weight=start_weight,
            flat_warmup_steps=0,
            end_weight=end_weight,
            total_steps=10,
        )


def test_estimate_total_steps() -> None:

    assert estimate_total_steps(dataset_size=1000, batch_size=10, max_epochs=5) == 500

    assert estimate_total_steps(dataset_size=1050, batch_size=20, max_epochs=3) == 159

    assert estimate_total_steps(dataset_size=1050, batch_size=20, max_epochs=3, drop_last=True) == 156

    assert estimate_total_steps(dataset_size=1000, batch_size=20, max_epochs=5, accumulate_grad_batches=2) * 2 == estimate_total_steps(dataset_size=1000, batch_size=20, max_epochs=5, accumulate_grad_batches=1)
    
    assert estimate_total_steps(dataset_size=1000, max_steps=42,batch_size=20, max_epochs=5, accumulate_grad_batches=2) == 42
    
    assert estimate_total_steps(dataset_size=1000, batch_size=10, max_epochs=5, num_nodes=4) * 4 == estimate_total_steps(dataset_size=1000, batch_size=10, max_epochs=5, num_nodes=1)
    
    assert estimate_total_steps(dataset_size=1000, batch_size=10, max_epochs=5, num_gpus_per_node=4) * 4 == estimate_total_steps(dataset_size=1000, batch_size=10, max_epochs=5, num_gpus_per_node=1)
    
    assert estimate_total_steps(dataset_size=1000, batch_size=10, max_epochs=5, limit_train_batches=0.5) == 250
    
    
    
    
    