import numpy as np
from omegaconf import OmegaConf

from scripts.evaluate_test_set import _ckpt_tag, _exclude_context


def test_exclude_context_removes_context_stations():
    stations = np.array(["a", "b", "c", "d"])
    cfg_split = OmegaConf.create({"test_context_station_indices": ["b", "d"]})
    result = _exclude_context([0, 1, 2, 3], cfg_split, stations)
    assert sorted(result) == [0, 2]


def test_exclude_context_noop_without_context_field():
    stations = np.array(["a", "b", "c", "d"])
    cfg_split = OmegaConf.create({})
    result = _exclude_context([0, 1, 2, 3], cfg_split, stations)
    assert sorted(result) == [0, 1, 2, 3]


def test_ckpt_tag_differs_with_and_without_exclude_context():
    """The output filename must differ between the two eval arms.

    Without this, a run with --exclude-context overwrites the CSV and
    summary from a run without it (and vice versa).
    """
    tag_with_context = _ckpt_tag("my_ckpt", exclude_context=False)
    tag_without_context = _ckpt_tag("my_ckpt", exclude_context=True)
    assert tag_with_context != tag_without_context
    assert tag_with_context == "my_ckpt"
    assert tag_without_context == "my_ckpt_nocontext"
