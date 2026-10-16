"""Panel E of the OAT scheme: predictive (posterior) distribution using real data.

Two panels over the same Bernese Oberland zoom as panels A and B:
  E1 — posterior mean  (NWP prior corrected by station innovations)
  E2 — posterior uncertainty  (altitude-based prior uncertainty reduced near stations)

Run: poetry run python -m scripts.plot_oat_panel_e_real
Output: data/figures/oat_panel_e_real.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from pyproj import Transformer

from oat.paths import data_root

from ._shared import canonical_stations, canonical_queries, canonical_queries_grid

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_COSMO_DIR = data_root() / "cosmo-1e_rechunked"
TOPO_PATH = _COSMO_DIR / "topo.zarr"
NWP_EXTRACTED_PATH = _COSMO_DIR / "nwp_extracted.zarr"
NWP_REF_PATH = _COSMO_DIR / "nwp.zarr"
ANEMOI_PATH = str(data_root() / "mch-co1e-an-archive-0p01-2019-2024-1h-v1-pl13.zarr")
OUT_PATH = Path(__file__).parents[3] / "data/figures/oat_panel_e_real.png"

# ---------------------------------------------------------------------------
# Style — identical to plot_oat_panel_b_real.py
# ---------------------------------------------------------------------------
ORANGE = "#E8732A"
GRAY_STA = "#AAAAAA"
BLUE_Q = "#3A72C6"
CMAP_FIELD = "RdBu_r"
FIELD_VLIM = (-3.5, 3.5)
CMAP_UNC = "YlOrBr"

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

STEP_ZOOM = 2
ZOOM_CENTER = (2_637_000.0, 1_163_000.0)
ZOOM_HALF_KM = 25_000.0
TIME_IDX = 5000
SEED = 42

plt.rcParams.update(
    {"font.family": "sans-serif", "font.size": 9, "axes.linewidth": 0.8}
)


# ---------------------------------------------------------------------------
# Data loaders  (identical to plot_oat_panel_b_real.py)
# ---------------------------------------------------------------------------
def _zoom_bounds() -> tuple[float, float, float, float]:
    cx, cy = ZOOM_CENTER
    return cx - ZOOM_HALF_KM, cx + ZOOM_HALF_KM, cy - ZOOM_HALF_KM, cy + ZOOM_HALF_KM


def load_topo_zoom() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def load_t2m_on_ref_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import xarray as xr
    from anemoi.datasets import open_dataset

    print("  Opening anemoi dataset …")
    anemoi_ds = open_dataset(ANEMOI_PATH)
    lon_a = anemoi_ds.longitudes
    lat_a = anemoi_ds.latitudes

    print("  Projecting to LV95 …")
    tf = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    east_a, north_a = tf.transform(lon_a, lat_a)

    ref = xr.open_zarr(str(NWP_REF_PATH))
    x_nwp = ref["x"].values
    y_nwp = ref["y"].values
    grid_e, grid_n = np.meshgrid(x_nwp, y_nwp)

    print("  Building KDTree …")
    tree = cKDTree(np.column_stack([east_a, north_a]))
    _, idx = tree.query(np.column_stack([grid_e.ravel(), grid_n.ravel()]))

    print("  Reading T_2M from extracted zarr …")
    store = zarr.open(str(NWP_EXTRACTED_PATH), mode="r")
    var_names = list(store.attrs["variables"])
    t2m_col = var_names.index("T_2M")
    t2m_flat = store[TIME_IDX, t2m_col, :]

    t2m_2d = t2m_flat[idx].reshape(640, 710)
    return t2m_2d.astype(np.float32), x_nwp, y_nwp


def load_t2m_zoom(
    t2m_nwp: np.ndarray, x_nwp: np.ndarray, y_nwp: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xlo, xhi, ylo, yhi = _zoom_bounds()
    xi0 = max(0, int(np.searchsorted(x_nwp, xlo)) - 1)
    xi1 = min(len(x_nwp), int(np.searchsorted(x_nwp, xhi)) + 2)
    yi0 = max(0, int(np.searchsorted(y_nwp, ylo)) - 1)
    yi1 = min(len(y_nwp), int(np.searchsorted(y_nwp, yhi)) + 2)
    return t2m_nwp[yi0:yi1, xi0:xi1], x_nwp[xi0:xi1], y_nwp[yi0:yi1]


def load_stations_split() -> tuple[np.ndarray, np.ndarray]:
    """Return (obs, missing) using the canonical split from _shared."""
    return canonical_stations()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _add_scale_bar(
    ax: plt.Axes, xhi: float, ylo: float, n_km: int = 5, bar_h: float = 380.0
) -> None:
    km = 1_000.0
    x_end = xhi - 2_000.0
    x_start = x_end - n_km * km
    y_bar = ylo + 1_400.0
    ax.add_patch(mpatches.FancyBboxPatch(
        (x_start - 700, y_bar - 1400), n_km * km + 1400, bar_h + 1650,
        boxstyle="round,pad=0", facecolor="white", edgecolor="none",
        alpha=0.75, zorder=6,
    ))
    for i in range(n_km):
        ax.add_patch(mpatches.Rectangle(
            (x_start + i * km, y_bar), km, bar_h,
            facecolor="black" if i % 2 == 0 else "white",
            edgecolor="black", linewidth=0.5, zorder=7,
        ))
    ax.text(
        x_start + n_km * km / 2, y_bar - 110,
        f"{n_km} km", fontsize=9, zorder=8, ha="center", va="top",
    )


def _draw_terrain(
    ax: plt.Axes, dem: np.ndarray, x_topo: np.ndarray, y_topo: np.ndarray
) -> None:
    ext = [x_topo[0], x_topo[-1], y_topo[0], y_topo[-1]]
    ls = LightSource(azdeg=310, altdeg=40)
    terrain_rgb = ls.shade(dem, cmap=TERRAIN_CMAP, vert_exag=2.5, blend_mode="soft")
    ax.imshow(terrain_rgb, origin="lower", aspect="equal",
              interpolation="bilinear", extent=ext)


def _scatter_stations(
    ax: plt.Axes, obs_sta: np.ndarray, miss_sta: np.ndarray, queries: np.ndarray
) -> None:
    ax.scatter(obs_sta[:, 0], obs_sta[:, 1],
               c=ORANGE, s=80, zorder=6, edgecolors="white", linewidths=0.8)
    ax.scatter(miss_sta[:, 0], miss_sta[:, 1],
               c=GRAY_STA, s=55, zorder=6, edgecolors="white", linewidths=0.8, alpha=0.85)
    ax.scatter(queries[:, 0], queries[:, 1],
               c=BLUE_Q, s=60, zorder=6, marker="D", edgecolors="white", linewidths=0.8)
    grid_queries = canonical_queries_grid()
    ax.scatter(grid_queries[:, 0], grid_queries[:, 1],
               c=BLUE_Q, s=60, zorder=6, marker="D", edgecolors="white", linewidths=0.8)


def _query_locations(xlo: float, xhi: float, ylo: float, yhi: float) -> np.ndarray:
    return canonical_queries()


def _set_ax(ax: plt.Axes, xlo: float, xhi: float, ylo: float, yhi: float) -> None:
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    _add_scale_bar(ax, xhi, ylo)


# ---------------------------------------------------------------------------
# Posterior field helpers
# ---------------------------------------------------------------------------
def _posterior_mean(
    prior: np.ndarray,
    x_topo: np.ndarray,
    y_topo: np.ndarray,
    obs_sta: np.ndarray,
    sigma: float = 5_000.0,
) -> np.ndarray:
    """Add synthetic innovations at observed station locations to the prior."""
    rng = np.random.RandomState(SEED + 3)
    innovations = rng.randn(len(obs_sta)) * 0.8
    YY, XX = np.meshgrid(y_topo, x_topo, indexing="ij")
    post = prior.copy()
    for (sx, sy), inn in zip(obs_sta, innovations):
        dist2 = (XX - sx) ** 2 + (YY - sy) ** 2
        post += inn * np.exp(-dist2 / (2 * sigma ** 2))
    return np.clip(post, *FIELD_VLIM)


def _posterior_uncertainty(
    dem: np.ndarray,
    x_topo: np.ndarray,
    y_topo: np.ndarray,
    obs_sta: np.ndarray,
    sigma: float = 6_000.0,
    smooth_sigma: float = 8.0,
) -> np.ndarray:
    """Altitude-based prior uncertainty reduced by Gaussian decay near each observed station."""
    smoothed = gaussian_filter(dem.astype(np.float32), sigma=smooth_sigma)
    lo, hi = smoothed.min(), smoothed.max()
    unc = (smoothed - lo) / (hi - lo + 1e-6)

    YY, XX = np.meshgrid(y_topo, x_topo, indexing="ij")
    for sx, sy in obs_sta:
        dist2 = (XX - sx) ** 2 + (YY - sy) ** 2
        unc -= 0.95 * np.exp(-dist2 / (2 * sigma ** 2))
    return np.clip(unc, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Panel plots
# ---------------------------------------------------------------------------
def plot_panel_e_mean(
    ax: plt.Axes,
    t2m_nwp: np.ndarray,
    x_nwp: np.ndarray,
    y_nwp: np.ndarray,
) -> None:
    """E1: posterior mean — NWP prior corrected by station innovations."""
    dem, x_topo, y_topo = load_topo_zoom()
    t2m_crop, x_nwp_z, y_nwp_z = load_t2m_zoom(t2m_nwp, x_nwp, y_nwp)

    mu = float(np.nanmean(t2m_nwp))
    sigma_t = float(np.nanstd(t2m_nwp))
    t2m_norm = np.clip((t2m_crop - mu) / max(sigma_t, 1e-6) * 1.5, *FIELD_VLIM)

    interp = RegularGridInterpolator(
        (y_nwp_z, x_nwp_z), t2m_norm, method="linear",
        bounds_error=False, fill_value=float(np.nanmean(t2m_norm)),
    )
    YY, XX = np.meshgrid(y_topo, x_topo, indexing="ij")
    prior_hires = interp(np.stack([YY.ravel(), XX.ravel()], axis=-1)).reshape(YY.shape)

    obs_sta, miss_sta = load_stations_split()
    post_mean = _posterior_mean(prior_hires, x_topo, y_topo, obs_sta)

    xlo, xhi, ylo, yhi = _zoom_bounds()
    ext_topo = [x_topo[0], x_topo[-1], y_topo[0], y_topo[-1]]

    _draw_terrain(ax, dem, x_topo, y_topo)

    norm_obj = plt.Normalize(vmin=FIELD_VLIM[0], vmax=FIELD_VLIM[1])
    post_rgba = plt.cm.RdBu_r(norm_obj(post_mean)).copy()
    post_rgba[:, :, 3] = 0.72
    ax.imshow(post_rgba, origin="lower", aspect="equal",
              interpolation="bilinear", extent=ext_topo)

    queries = _query_locations(xlo, xhi, ylo, yhi)
    _scatter_stations(ax, obs_sta, miss_sta, queries)
    _set_ax(ax, xlo, xhi, ylo, yhi)


def plot_panel_e_uncertainty(ax: plt.Axes) -> None:
    """E2: posterior uncertainty — altitude-based prior reduced near observed stations."""
    dem, x_topo, y_topo = load_topo_zoom()
    obs_sta, miss_sta = load_stations_split()

    xlo, xhi, ylo, yhi = _zoom_bounds()
    ext_topo = [x_topo[0], x_topo[-1], y_topo[0], y_topo[-1]]

    _draw_terrain(ax, dem, x_topo, y_topo)

    unc = _posterior_uncertainty(dem, x_topo, y_topo, obs_sta)
    unc_norm = plt.Normalize(vmin=0.0, vmax=1.0)
    unc_rgba = plt.colormaps[CMAP_UNC](unc_norm(unc)).copy()
    unc_rgba[:, :, 3] = 0.72
    ax.imshow(unc_rgba, origin="lower", aspect="equal",
              interpolation="bilinear", extent=ext_topo)

    queries = _query_locations(xlo, xhi, ylo, yhi)
    _scatter_stations(ax, obs_sta, miss_sta, queries)
    _set_ax(ax, xlo, xhi, ylo, yhi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading T_2M from NWP …")
    t2m_nwp, x_nwp, y_nwp = load_t2m_on_ref_grid()

    print("Plotting …")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor="white")
    plot_panel_e_mean(ax1, t2m_nwp, x_nwp, y_nwp)
    plot_panel_e_uncertainty(ax2)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PATH.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_PATH.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
