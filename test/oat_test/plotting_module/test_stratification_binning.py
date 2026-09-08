import numpy as np
import pandas as pd

from oat.plotting_module.stratification import _bin_keys


def test_season_keys_canonical_order():
    df = pd.DataFrame({"season": ["JJA", "DJF", "SON", "DJF", "MAM"]})
    keys, present, _label_for, order_for = _bin_keys(df, "season", nbins=5)
    # Only present seasons, in canonical DJF/MAM/JJA/SON order.
    assert present == ["DJF", "MAM", "JJA", "SON"]
    assert [order_for[s] for s in present] == [0, 1, 2, 3]
    assert list(keys) == ["JJA", "DJF", "SON", "DJF", "MAM"]


def test_continuous_fixed_width_bins():
    df = pd.DataFrame({"nearest_obs_dist": [1.0, 1.5, 9.0, 9.5]})
    keys, present, label_for, _order = _bin_keys(df, "nearest_obs_dist", nbins=2)
    # Two fixed-width bins: {1.0,1.5} -> bin 0, {9.0,9.5} -> bin 1.
    assert present == [0, 1]
    assert list(keys) == [0, 0, 1, 1]
    assert len({label_for[b] for b in present}) == 2


def test_elevation_custom_bins():
    # Fixed, hand-chosen bins: <750, 750-1250, 1250-2000, >2000.
    df = pd.DataFrame({"elevation": [500.0, 1000.0, 1800.0, 2500.0]})
    keys, present, label_for, _order = _bin_keys(df, "elevation", nbins=5)
    assert present == [0, 1, 2, 3]
    assert list(keys) == [0, 1, 2, 3]
    assert [label_for[b] for b in present] == [
        "<750",
        "750-1250",
        "1250-2000",
        ">2000",
    ]


def test_empty_dataframe_returns_empty():
    df = pd.DataFrame({"elevation": []})
    keys, present, label_for, order_for = _bin_keys(df, "elevation", nbins=5)
    assert present == []
    assert label_for == {} and order_for == {}
    assert keys.size == 0


def test_all_nan_stratifier_returns_empty():
    df = pd.DataFrame({"elevation": [np.nan, np.nan, np.nan]})
    _keys, present, label_for, order_for = _bin_keys(df, "elevation", nbins=5)
    assert present == []
    assert label_for == {} and order_for == {}


def test_nan_values_excluded_not_folded_into_last_bin():
    # Regression: np.digitize maps NaN to the last bin via clip. NaN covariate
    # rows must instead get the sentinel key -1 and be excluded from every bin.
    df = pd.DataFrame({"elevation": [100.0, 200.0, 300.0, 400.0, np.nan]})
    keys, present, _label_for, _order = _bin_keys(df, "elevation", nbins=2)
    # The NaN row (last) must NOT be assigned to any real bin.
    assert keys[-1] == -1
    # Bins present come only from the four finite values.
    assert set(present).issubset({0, 1})
    # No real bin key equals the NaN sentinel.
    assert -1 not in present
    # Every finite value landed in a real (non-sentinel) bin.
    assert all(keys[i] in present for i in range(4))
