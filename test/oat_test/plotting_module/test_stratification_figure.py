import matplotlib

matplotlib.use("Agg")

import matplotlib.container
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch

from oat.plotting_module.stratification import (
    STRATIFIERS,
    _bin_keys,
    make_stratification_figure,
)


def _long_df():
    rng = np.random.default_rng(0)
    rows = []
    seasons = ["DJF", "MAM", "JJA", "SON"]
    for model in ["oat", "gp"]:
        for i in range(200):
            rows.append(
                {
                    "model": model,
                    "season": seasons[i % 4],
                    "elevation": 100.0 + 10.0 * (i % 50),
                    "nearest_obs_dist": 1.0 + 0.1 * (i % 30),
                    "lon": 6.0 + 0.01 * (i % 40),
                    "lat": 46.0 + 0.01 * (i % 40),
                    "crps_temperature": float(rng.random()),
                    "crps_precipitation": float(rng.random()),
                }
            )
    return pd.DataFrame(rows)


def test_figure_grid_shape():
    df = _long_df()
    fig = make_stratification_figure(
        df,
        var_names=["temperature", "precipitation"],
        metric="crps",
        model_order=["oat", "gp"],
        nbins=4,
    )
    # rows = 2 variables, cols = 5 stratifiers
    assert len(fig.axes) >= 2 * len(STRATIFIERS)
    assert STRATIFIERS == ["season", "elevation", "nearest_obs_dist", "lon", "lat"]

    # Panels must draw boxplot boxes (PathPatch, since patch_artist=True), not bars.
    assert any(isinstance(p, PathPatch) for ax in fig.axes for p in ax.patches)
    assert not any(
        isinstance(c, matplotlib.container.BarContainer)
        for ax in fig.axes
        for c in ax.containers
    )

    import matplotlib.pyplot as plt

    plt.close(fig)


def _narrow_lat_df():
    # Latitude over Switzerland spans ~45.8-47.8: a sub-2-unit range that, with
    # `:.0f` bin-label formatting, would collapse to duplicate integer labels.
    rng = np.random.default_rng(1)
    seasons = ["DJF", "MAM", "JJA", "SON"]
    rows = []
    for model in ["oat", "gp"]:
        for i in range(100):
            rows.append(
                {
                    "model": model,
                    "season": seasons[i % 4],
                    "elevation": 100.0 + 10.0 * (i % 50),
                    "nearest_obs_dist": 1.0 + 0.1 * (i % 30),
                    "lon": 6.0 + 0.01 * (i % 40),
                    "lat": 45.8 + 2.0 * (i / 99),
                    "crps_temperature": float(rng.random()),
                }
            )
    return pd.DataFrame(rows)


def test_narrow_range_binning_does_not_drop_bars():
    df = _narrow_lat_df()
    _keys, present, label_for, _order = _bin_keys(df, "lat", nbins=5)
    # Distinct bins must all get distinct labels (no silent merge to one bar).
    assert len(present) == 5
    assert len({label_for[b] for b in present}) == 5

    fig = make_stratification_figure(
        df,
        var_names=["temperature"],
        metric="crps",
        model_order=["oat", "gp"],
        nbins=5,
    )
    import matplotlib.pyplot as plt

    lat_ax = fig.axes[STRATIFIERS.index("lat")]
    assert len(lat_ax.get_xticks()) == 5
    plt.close(fig)
