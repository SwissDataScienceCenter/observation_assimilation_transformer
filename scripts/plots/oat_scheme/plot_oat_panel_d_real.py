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
from matplotlib.colors import LinearSegmentedColormap, LightSource, ListedColormap
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase

from oat.paths import data_root

from ._shared import canonical_stations, canonical_queries, canonical_queries_grid

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_COSMO_DIR = data_root() / "cosmo-1e_rechunked"
TOPO_PATH = _COSMO_DIR / "topo.zarr"
OUT_PATH = Path(__file__).parents[3] / "data/figures/oat_panel_d_real.png"

# ---------------------------------------------------------------------------
# Style — identical to plot_oat_panel_a_real.py
# ---------------------------------------------------------------------------
ORANGE = "#E8732A"
ORANGE_L = "#F5C080"
GRAY_STA = "#AAAAAA"
BLUE_Q = "#3A72C6"
BLUE_L = "#A0BEEC"
# Cross-attention arrows — a distinct colour from the query blue (BLUE_Q)
# and the self-attention orange, so the two attention types read separately.
# Dark grey (darker than GRAY_STA so it's not confused with missing stations).
CROSS_ATTN = "#656565"

# TERRAIN_CMAP = LinearSegmentedColormap.from_list(
#     "terrain_oat",
#     [
#         (0.00, "#3B5E1F"),
#         (0.12, "#5C8A3A"),
#         (0.28, "#8EAA62"),
#         (0.42, "#C8B57A"),
#         (0.55, "#96703C"),
#         (0.68, "#7A6050"),
#         (0.80, "#A09898"),
#         (0.90, "#D2D2D2"),
#         (1.00, "#F4F4F4"),
#     ],
# )


def _mute_cmap(cmap, strength=0.5, toward=(0.9, 0.9, 0.9, 1.0)):
    """strength=1.0 → original, 0.0 → flat grey."""
    cmap = plt.get_cmap(cmap)
    cols = cmap(np.linspace(0, 1, 256))
    cols = strength * cols + (1 - strength) * np.array(toward)
    return ListedColormap(cols)


def _crop_cmap(cmap, lo=0.0, hi=1.0, n=256):
    """Use only the [lo, hi] slice of a colormap (0=bottom, 1=top).

    Raise `lo` to trim gist_earth's dark navy "sea level" band so the terrain
    starts from a paler blue/green instead of the deep blue.
    """
    cmap = plt.get_cmap(cmap)
    return ListedColormap(cmap(np.linspace(lo, hi, n)))


TERRAIN_CMAP = _mute_cmap(_crop_cmap("gist_earth", lo=0.15, hi=1.0), strength=0.8)

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
    """Return (observed, missing) using the canonical split from _shared.

    Synthetic stations pad the observed set to 12 so attention edges read clearly.
    """
    xlo, xhi, ylo, yhi = _zoom_bounds()
    real_obs, missing = canonical_stations()

    rng_synth = np.random.RandomState(SEED)
    n_synth = max(0, 12 - len(real_obs))
    if n_synth > 0:
        synth = np.column_stack([
            rng_synth.uniform(xlo + 3000, xhi - 3000, n_synth),
            rng_synth.uniform(ylo + 3000, yhi - 3000, n_synth),
        ])
        real_obs = np.vstack([real_obs, synth])

    return real_obs, missing


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
    ax: plt.Axes, stations: np.ndarray, queries: np.ndarray, k: int = 4
) -> None:
    """Arrows from each query's k nearest stations to that query (information propagation)."""
    for qi in range(len(queries)):
        dists = np.linalg.norm(stations - queries[qi], axis=1)
        for rank, si in enumerate(np.argsort(dists)[:k]):
            w = _decay_weight(queries[qi], stations[si])
            alpha = (0.55 + w * 0.40) if rank == 0 else (0.20 + w * 0.45)
            lw = (1.0 + w * 2.0) if rank == 0 else (0.5 + w * 1.5)
            ax.annotate(
                "",
                xy=(queries[qi, 0], queries[qi, 1]),
                xytext=(stations[si, 0], stations[si, 1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=CROSS_ATTN,
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
    # ax.set_title(
    #     "1) Station self-attention\n(spatial mixing among observations)", fontsize=9
    # )


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
    grid_queries = canonical_queries_grid()
    all_queries = np.vstack([queries, grid_queries])
    _draw_terrain(ax, dem, x_1d, y_1d)
    _draw_cross_attn_arrows(ax, all_sta, all_queries)
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
    ax.scatter(
        grid_queries[:, 0], grid_queries[:, 1],
        c=BLUE_Q, s=60, zorder=8, marker="D", edgecolors="white", linewidths=0.8,
    )
    # ax.set_title(
    #     "2) Cross-attention to query locations\n(propagate information)", fontsize=9
    # )


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
class _ArrowHandler(HandlerBase):
    """Render a short arrow as a legend handle (for cross-attention)."""

    def __init__(self, color: str, lw: float = 2.2) -> None:
        self._color = color
        self._lw = lw
        super().__init__()

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):  # noqa: E501
        y = ydescent + height / 2.0
        arrow = FancyArrowPatch(
            (xdescent, y), (xdescent + width, y),
            arrowstyle="-|>", mutation_scale=13, color=self._color, lw=self._lw,
        )
        arrow.set_transform(trans)
        return [arrow]


def _add_bottom_legend(fig: plt.Figure) -> None:
    """Figure-level legend at the bottom, matching panel A's style."""
    marker = dict(color="none", markeredgecolor="white", markeredgewidth=0.8)
    arrow_proxy = Line2D([0], [0], color=CROSS_ATTN, lw=2.5, label="Cross-attention")
    handles = [
        # Line2D([0], [0], marker="o", markerfacecolor=ORANGE, markersize=11,
        #        label="Observed station", **marker),
        # Line2D([0], [0], marker="o", markerfacecolor=GRAY_STA, markersize=9,
        #        label="Missing observation", **marker),
        # Line2D([0], [0], marker="D", markerfacecolor=BLUE_Q, markersize=9,
        #        label="Query location", **marker),
        Line2D([0], [0], color=ORANGE, lw=2.8, label="Self-attention"),
        arrow_proxy,
    ]
    leg = fig.legend(
        handles=handles,
        handler_map={arrow_proxy: _ArrowHandler(CROSS_ATTN)},
        loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=5,
        fontsize=12, framealpha=0.92, edgecolor="#CCCCCC",
        handlelength=2.2, columnspacing=1.8, borderpad=0.8,
    )
    leg.set_zorder(10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading topography …")
    dem, x_1d, y_1d = load_topo_zoom()

    print("Loading stations …")
    obs, missing = load_stations_split()
    queries = canonical_queries()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 11), facecolor="white")

    plot_self_attn_panel(ax1, dem, x_1d, y_1d, obs, missing)
    plot_cross_attn_panel(ax2, dem, x_1d, y_1d, obs, missing, queries)

    fig.tight_layout(pad=1.5)
    _add_bottom_legend(fig)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PATH.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_PATH.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
