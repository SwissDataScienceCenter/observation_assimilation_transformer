import numpy as np
import pandas as pd
import pytest

from oat.plotting_module.stratification import stratified_bar_stats


def _df():
    # Two models, one continuous covariate, one categorical season.
    rows = []
    for model, base in [("oat", 1.0), ("gp", 2.0)]:
        for elev, season, val in [
            (100.0, "DJF", base + 0.0),
            (150.0, "DJF", base + 0.2),
            (900.0, "JJA", base + 1.0),
            (950.0, "JJA", base + 1.2),
        ]:
            rows.append(
                {"model": model, "elevation": elev, "season": season, "crps_t": val}
            )
    return pd.DataFrame(rows)


def test_season_stats_ordering_and_means():
    stats = stratified_bar_stats(_df(), "season", "crps_t", nbins=5)
    # season bins present in canonical order, DJF before JJA
    djf = stats[(stats["bin"] == "DJF") & (stats["model"] == "oat")]
    assert djf["mean"].iloc[0] == pytest.approx(1.1)  # mean(1.0, 1.2)
    assert list(stats.sort_values("bin_order")["bin"].unique()) == ["DJF", "JJA"]


def test_continuous_binning_counts():
    stats = stratified_bar_stats(_df(), "elevation", "crps_t", nbins=2)
    # 2 elevation bins x 2 models = 4 rows, each aggregating 2 samples
    assert len(stats) == 4
    assert set(stats["n"]) == {2}


def test_nan_metric_ignored():
    df = _df()
    df.loc[df.index[0], "crps_t"] = np.nan
    stats = stratified_bar_stats(df, "season", "crps_t", nbins=5)
    row = stats[(stats["bin"] == "DJF") & (stats["model"] == "oat")]
    assert row["n"].iloc[0] == 1  # one NaN dropped


def test_empty_dataframe_returns_empty_stats():
    empty = pd.DataFrame(
        columns=["model", "elevation", "season", "crps_t"]
    )
    out = stratified_bar_stats(empty, "elevation", "crps_t", nbins=5)
    assert len(out) == 0
    assert list(out.columns) == ["bin", "bin_order", "model", "mean", "se", "n"]

    out2 = stratified_bar_stats(empty, "season", "crps_t", nbins=5)
    assert len(out2) == 0
    assert list(out2.columns) == ["bin", "bin_order", "model", "mean", "se", "n"]
