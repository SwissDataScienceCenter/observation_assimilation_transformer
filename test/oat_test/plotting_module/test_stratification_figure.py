import matplotlib

matplotlib.use("Agg")

import matplotlib.container
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch

from oat.plotting_module.stratification import (
    STRATIFIERS,
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
                    "station_x": 6.0 + 0.01 * (i % 40),
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
    # rows = 2 variables, cols = 3 stratifiers
    assert len(fig.axes) >= 2 * len(STRATIFIERS)
    assert STRATIFIERS == ["season", "elevation", "nearest_obs_dist"]

    # Panels must draw boxplot boxes (PathPatch, since patch_artist=True), not bars.
    assert any(isinstance(p, PathPatch) for ax in fig.axes for p in ax.patches)
    assert not any(
        isinstance(c, matplotlib.container.BarContainer)
        for ax in fig.axes
        for c in ax.containers
    )

    import matplotlib.pyplot as plt

    plt.close(fig)
