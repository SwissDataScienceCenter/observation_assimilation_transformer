"""Panel D of the OAT scheme: station self-attention and cross-attention.

Two 2D panels over the same Bernese Oberland zoom area as panel A.

Run: poetry run python -m scripts.plot_oat_panel_d_real
Output: data/figures/oat_panel_d_real.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, LightSource
from matplotlib.patches import FancyArrowPatch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TOPO_PATH = Path(
    "/scratch/mch/cdonner/data/obs_downscaling/raw/cosmo-1e_rechunked/topo.zarr"
)
OUT_PATH = Path(__file__).parents[3] / "data/figures/oat_panel_d_real.png"

# ---------------------------------------------------------------------------
# Style — identical to plot_oat_panel_a_real.py
# ---------------------------------------------------------------------------
ORANGE = "#E8732A"
ORANGE_L = "#F5C080"
GRAY_STA = "#AAAAAA"
BLUE_Q = "#3A72C6"
BLUE_L = "#A0BEEC"

TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    "terrain_oat",
    [
        (0.00, "#3B5E1F"),
        (0.12, "#5C8A3A"),
        (0.28, "#8EAA62"),
        (0.42, "#C8B57A"),
        (0.55, "#96703C"),
        (0.68, "#7A6050"),
        (0.80, "#A09898"),
        (0.90, "#D2D2D2"),
        (1.00, "#F4F4F4"),
    ],
)

SEED = 42
STEP_ZOOM = 2       # 100 m resolution — same as plot_oat_panel_a_real.py zoom panel
ZOOM_CENTER = (2_637_000.0, 1_163_000.0)
ZOOM_HALF_KM = 25_000.0

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.linewidth": 0.8})


# ---------------------------------------------------------------------------
# Data loaders  (identical to plot_oat_panel_a_real.py)
# ---------------------------------------------------------------------------
def _zoom_bounds() -> tuple[float, float, float, float]:
    cx, cy = ZOOM_CENTER
    return cx - ZOOM_HALF_KM, cx + ZOOM_HALF_KM, cy - ZOOM_HALF_KM, cy + ZOOM_HALF_KM


def load_topo_zoom() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dem, x_1d, y_1d) cropped to zoom window at STEP_ZOOM resolution."""
    xlo, xhi, ylo, yhi = _zoom_bounds()
    z = zarr.open(str(TOPO_PATH), mode="r")
    x_full = z["x"][:]
    y_full = z["y"][:]
    xi0, xi1 = int(np.searchsorted(x_full, xlo)), int(np.searchsorted(x_full, xhi))
    yi0, yi1 = int(np.searchsorted(y_full, ylo)), int(np.searchsorted(y_full, yhi))
    dem = z["DEM"][yi0:yi1:STEP_ZOOM, xi0:xi1:STEP_ZOOM]
    x_1d = x_full[xi0:xi1:STEP_ZOOM]
    y_1d = y_full[yi0:yi1:STEP_ZOOM]
    return dem.astype(np.float32), x_1d.astype(np.float64), y_1d.astype(np.float64)


def load_stations_split() -> tuple[np.ndarray, np.ndarray]:
    """Return (observed, missing) matching Panel A's zoom panel exactly.

    Real stations are loaded and split with the same SEED+1 logic as
    plot_oat_panel_a_real.py. Synthetic stations pad the observed set to 12.
    """
    from peakweather.dataset import PeakWeatherDataset

    xlo, xhi, ylo, yhi = _zoom_bounds()
    obs_ds = PeakWeatherDataset(
        root="/scratch/mch/cdonner/data/peakweather",
        station_type="meteo_station",
        freq="1h",
    )
    tbl = obs_ds.stations_table
    mask = (
        (tbl["swiss_easting"] >= xlo) & (tbl["swiss_easting"] <= xhi)
        & (tbl["swiss_northing"] >= ylo) & (tbl["swiss_northing"] <= yhi)
    )
    real = tbl.loc[mask, ["swiss_easting", "swiss_northing"]].values

    # Identical to plot_oat_panel_a_real.py zoom panel
    rng_miss = np.random.RandomState(SEED + 1)
    missing_idx = rng_miss.randint(len(real))
    obs_mask = np.ones(len(real), dtype=bool)
    obs_mask[missing_idx] = False
    real_obs = real[obs_mask]
    missing = real[~obs_mask]

    # Pad observed set with synthetic stations so self/cross-attention reads clearly
    rng_synth = np.random.RandomState(SEED)
    n_synth = max(0, 12 - len(real_obs))
    if n_synth > 0:
        synth = np.column_stack([
            rng_synth.uniform(xlo + 3000, xhi - 3000, n_synth),
            rng_synth.uniform(ylo + 3000, yhi - 3000, n_synth),
        ])
        real_obs = np.vstack([real_obs, synth])

    return real_obs, missing


def _invented_queries(all_stations: np.ndarray, n: int = 10) -> np.ndarray:
    """Random query locations that avoid all station positions (obs + missing)."""
    xlo, xhi, ylo, yhi = _zoom_bounds()
    rng = np.random.RandomState(SEED + 7)
    pts: list[np.ndarray] = []
    while len(pts) < n:
        xy = np.array([
            rng.uniform(xlo + 2000, xhi - 2000),
            rng.uniform(ylo + 2000, yhi - 2000),
        ])
        if np.min(np.linalg.norm(all_stations - xy, axis=1)) > 3000:
            pts.append(xy)
    return np.array(pts)


# ---------------------------------------------------------------------------
# Terrain rendering
# ---------------------------------------------------------------------------
def _make_vignette(h: int, w: int, edge_frac: float = 0.30) -> np.ndarray:
    u_x = np.linspace(0, 1, w)
    u_y = np.linspace(0, 1, h)
    XX, YY = np.meshgrid(u_x, u_y)

    def _step(v: np.ndarray) -> np.ndarray:
        t = np.clip(v / edge_frac, 0, 1)
        return t * t * (3 - 2 * t)

    return _step(XX) * _step(1 - XX) * _step(YY) * _step(1 - YY)


def _draw_terrain(ax: plt.Axes, dem: np.ndarray, x_1d: np.ndarray, y_1d: np.ndarray) -> None:
    """Render hillshaded terrain identical to plot_oat_panel_a_real.py zoom panel."""
    ext = [x_1d[0], x_1d[-1], y_1d[0], y_1d[-1]]
    ls = LightSource(azdeg=310, altdeg=40)
    terrain_rgb = ls.shade(dem, cmap=TERRAIN_CMAP, vert_exag=2.5, blend_mode="soft")
    terrain_rgb[..., :3] = terrain_rgb[..., :3] * 0.60 + 0.40  # desaturate toward white
    ax.imshow(terrain_rgb, origin="lower", aspect="equal",
              interpolation="bilinear", extent=ext)

    # Subtle vignette overlay to soften edges
    vig = _make_vignette(*dem.shape)
    vig_rgba = np.zeros((*dem.shape, 4))
    vig_rgba[..., 3] = (1 - vig) * 0.18
    ax.imshow(vig_rgba, origin="lower", aspect="equal",
              interpolation="bilinear", extent=ext, zorder=2)

    xlo, xhi, ylo, yhi = _zoom_bounds()
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


# ---------------------------------------------------------------------------
# Attention edge helpers
# ---------------------------------------------------------------------------
def _knn_edges(pos: np.ndarray, k: int) -> list[tuple[int, int]]:
    edges = []
    for i in range(len(pos)):
        dists = np.linalg.norm(pos - pos[i], axis=1)
        dists[i] = np.inf
        for j in np.argsort(dists)[:k]:
            edges.append((i, int(j)))
    return edges


def _decay_weight(pi: np.ndarray, pj: np.ndarray, sigma: float = 8000.0) -> float:
    d = float(np.linalg.norm(pi - pj))
    return float(np.exp(-(d**2) / (2 * sigma**2)))


def _draw_self_attn_edges(ax: plt.Axes, stations: np.ndarray) -> None:
    """KNN edges between stations, width/alpha encode attention weight."""
    edges = _knn_edges(stations, k=4)
    for i, j in edges:
        w = _decay_weight(stations[i], stations[j])
        if w < 0.04:
            continue
        ax.plot(
            [stations[i, 0], stations[j, 0]],
            [stations[i, 1], stations[j, 1]],
            color=ORANGE,
            lw=0.8 + w * 2.8,
            alpha=0.35 + w * 0.55,
            zorder=3,
            solid_capstyle="round",
        )


def _draw_cross_attn_arrows(
    ax: plt.Axes, stations: np.ndarray, queries: np.ndarray, k: int = 3
) -> None:
    """Arrows from each station to its k nearest queries (information propagation).

    Every station gets at least one arrow; weight encodes distance decay.
    """
    for si in range(len(stations)):
        dists = np.linalg.norm(queries - stations[si], axis=1)
        for rank, qi in enumerate(np.argsort(dists)[:k]):
            w = _decay_weight(stations[si], queries[qi])
            # Closest query always visible; farther ones can be lighter
            alpha = (0.55 + w * 0.40) if rank == 0 else (0.20 + w * 0.45)
            lw = (1.0 + w * 2.0) if rank == 0 else (0.5 + w * 1.5)
            ax.annotate(
                "",
                xy=(queries[qi, 0], queries[qi, 1]),
                xytext=(stations[si, 0], stations[si, 1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=BLUE_Q,
                    lw=lw,
                    alpha=alpha,
                    mutation_scale=14,
                ),
                zorder=3,
            )


# ---------------------------------------------------------------------------
# Panel plots
# ---------------------------------------------------------------------------
def _scatter_missing(ax: plt.Axes, missing: np.ndarray) -> None:
    if len(missing) == 0:
        return
    ax.scatter(
        missing[:, 0], missing[:, 1],
        c=GRAY_STA, s=55, zorder=6, edgecolors="white", linewidths=0.8, alpha=0.85,
    )


def plot_self_attn_panel(
    ax: plt.Axes,
    dem: np.ndarray,
    x_1d: np.ndarray,
    y_1d: np.ndarray,
    obs: np.ndarray,
    missing: np.ndarray,
) -> None:
    all_sta = np.vstack([obs, missing])
    _draw_terrain(ax, dem, x_1d, y_1d)
    _draw_self_attn_edges(ax, all_sta)
    _scatter_missing(ax, missing)
    ax.scatter(
        obs[:, 0], obs[:, 1],
        c=ORANGE, s=80, zorder=7, edgecolors="white", linewidths=0.8,
    )
    ax.set_title(
        "1) Station self-attention\n(spatial mixing among observations)", fontsize=9
    )


def plot_cross_attn_panel(
    ax: plt.Axes,
    dem: np.ndarray,
    x_1d: np.ndarray,
    y_1d: np.ndarray,
    obs: np.ndarray,
    missing: np.ndarray,
    queries: np.ndarray,
) -> None:
    all_sta = np.vstack([obs, missing])
    _draw_terrain(ax, dem, x_1d, y_1d)
    _draw_cross_attn_arrows(ax, all_sta, queries, k=3)
    _scatter_missing(ax, missing)
    ax.scatter(
        obs[:, 0], obs[:, 1],
        c=ORANGE, s=80, zorder=7, edgecolors="white", linewidths=0.8,
    )
    ax.scatter(
        missing[:, 0], missing[:, 1],
        c=GRAY_STA, s=55, zorder=6, edgecolors="white", linewidths=0.8, alpha=0.85,
    ) if len(missing) > 0 else None
    ax.scatter(
        queries[:, 0], queries[:, 1],
        c=BLUE_Q, s=60, zorder=8, marker="D", edgecolors="white", linewidths=0.8,
    )
    ax.set_title(
        "2) Cross-attention to query locations\n(propagate information)", fontsize=9
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading topography …")
    dem, x_1d, y_1d = load_topo_zoom()

    print("Loading stations …")
    obs, missing = load_stations_split()
    queries = _invented_queries(np.vstack([obs, missing]), n=10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5), facecolor="white")

    plot_self_attn_panel(ax1, dem, x_1d, y_1d, obs, missing)
    plot_cross_attn_panel(ax2, dem, x_1d, y_1d, obs, missing, queries)

    fig.tight_layout(pad=1.5)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
