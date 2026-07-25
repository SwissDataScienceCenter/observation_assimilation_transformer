"""Panel A of the OAT scheme using real COSMO-PeakWeather data.

Run: poetry run python -m scripts.plot_oat_panel_a_real
Output: data/figures/oat_panel_a_real.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase
from matplotlib.transforms import Bbox, TransformedBbox
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from pyproj import Transformer

from ._shared import canonical_stations, canonical_queries, canonical_queries_grid

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TOPO_PATH = Path(
    "/scratch/mch/cdonner/data/obs_downscaling/raw/cosmo-1e_rechunked/topo.zarr"
)
NWP_EXTRACTED_PATH = Path(
    "/scratch/mch/cdonner/data/obs_downscaling/raw/cosmo-1e_rechunked/nwp_extracted.zarr"
)
NWP_REF_PATH = Path(
    "/scratch/mch/cdonner/data/obs_downscaling/raw/cosmo-1e_rechunked/nwp.zarr"
)
ANEMOI_PATH = (
    "/store_new/mch/msopr/ml/datasets/mch-co1e-an-archive-0p01-2019-2024-1h-v1-pl13.zarr"
)
OUT_PATH = Path(__file__).parents[3] / "data/figures/oat_panel_a_real.png"

# ---------------------------------------------------------------------------
# Style — identical to plot_oat_scheme.py
# ---------------------------------------------------------------------------
ORANGE = "#E8732A"
GRAY_STA = "#AAAAAA"
BLUE_Q = "#3A72C6"
# NWP field colormap — change here to restyle every NWP overlay + colorbar.
# The field is a normalized anomaly around zero, so a diverging map usually
# reads best (e.g. "RdBu_r", "coolwarm", "RdYlBu_r", "bwr"), but any cmap works.
CMAP_FIELD = "RdYlBu_r"
FIELD_CMAP = plt.get_cmap(CMAP_FIELD)
FIELD_VLIM = (-3.5, 3.5)

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
# TERRAIN_CMAP = plt.cm.terrain
from matplotlib.colors import ListedColormap

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

class _PatchImageHandler(HandlerBase):
    """Render a small raster image as a legend handle.

    yoffset shifts the box down within its legend row (in fontsize units),
    so it doesn't bleed into the entry above.
    """

    def __init__(self, img: np.ndarray, interp: str = "nearest", yoffset: float = 0.0) -> None:
        self._img = img
        self._interp = interp
        self._yoffset = yoffset
        super().__init__()

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):  # noqa: E501
        shift = self._yoffset * fontsize
        bb = Bbox.from_bounds(xdescent, ydescent - shift, width, height)
        img = BboxImage(TransformedBbox(bb, trans), interpolation=self._interp, zorder=3)
        img.set_data(self._img)
        return [img]


STEP = 10  # subsample every STEP-th topo pixel → 500m resolution
STEP_ZOOM = 2  # for zoom: 100m resolution
TIME_IDX = 5000  # a summer-ish timestep (adjust if missing-data gap)
SEED = 42

# Zoom window: Bernese Oberland (Interlaken → Jungfrau), 50×50 km in LV95
ZOOM_CENTER = (2_637_000.0, 1_163_000.0)
ZOOM_HALF_KM = 25_000.0  # metres

# Zoom NWP normalization:
#   False → use full-domain mean/std (colours comparable to the full panel, but
#           the small window shows little contrast).
#   True  → standardize with the zoom crop's own mean/std, so local variation
#           fills the colour range (more visible change, not comparable to full).
ZOOM_LOCAL_NORM = True
ZOOM_CONTRAST = 2.0  # gain before clipping to FIELD_VLIM; raise for more contrast

plt.rcParams.update(
    {"font.family": "sans-serif", "font.size": 9, "axes.linewidth": 0.8}
)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_topo(step: int = STEP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dem, x_1d, y_1d) subsampled by `step`."""
    z = zarr.open(str(TOPO_PATH), mode="r")
    dem = z["DEM"][::step, ::step]  # (H', W')
    x_1d = z["x"][::step]  # (W',) — easting
    y_1d = z["y"][::step]  # (H',) — northing
    return dem.astype(np.float32), x_1d.astype(np.float64), y_1d.astype(np.float64)


def load_t2m_on_ref_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t2m_2d, x_nwp, y_nwp) on the 640×710 ref LV95 grid."""
    import xarray as xr
    from anemoi.datasets import open_dataset

    # 1. Get anemoi lat/lon for the flat 919 620-point grid
    print("  Opening anemoi dataset …")
    anemoi_ds = open_dataset(ANEMOI_PATH)
    lon_a = anemoi_ds.longitudes  # (919620,)
    lat_a = anemoi_ds.latitudes  # (919620,)

    # 2. Project anemoi grid to LV95
    print("  Projecting to LV95 …")
    tf = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    east_a, north_a = tf.transform(lon_a, lat_a)

    # 3. Ref NWP grid (regular LV95)
    ref = xr.open_zarr(str(NWP_REF_PATH))
    x_nwp = ref["x"].values  # (710,)
    y_nwp = ref["y"].values  # (640,)
    grid_e, grid_n = np.meshgrid(x_nwp, y_nwp)  # (640, 710)

    # 4. KDTree nearest-neighbour mapping anemoi → ref grid
    print("  Building KDTree …")
    tree = cKDTree(np.column_stack([east_a, north_a]))
    _, idx = tree.query(np.column_stack([grid_e.ravel(), grid_n.ravel()]))

    # 5. Read T_2M from extracted zarr for one timestep
    print("  Reading T_2M from extracted zarr …")
    store = zarr.open(str(NWP_EXTRACTED_PATH), mode="r")  # (45378, 6, 919620)
    var_names = list(store.attrs["variables"])
    t2m_col = var_names.index("T_2M")
    t2m_flat = store[TIME_IDX, t2m_col, :]  # (919620,)

    t2m_2d = t2m_flat[idx].reshape(640, 710)  # (640, 710)
    return t2m_2d.astype(np.float32), x_nwp, y_nwp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_vignette(h: int, w: int, edge_frac: float = 0.18) -> np.ndarray:
    u_x = np.linspace(0, 1, w)
    u_y = np.linspace(0, 1, h)
    XX, YY = np.meshgrid(u_x, u_y)

    def _step(v: np.ndarray) -> np.ndarray:
        t = np.clip(v / edge_frac, 0, 1)
        return t * t * (3 - 2 * t)

    return _step(XX) * _step(1 - XX) * _step(YY) * _step(1 - YY)


def _invented_stations(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(SEED)
    # Switzerland proper (tighter box inside the full COSMO domain)
    xlo, xhi = max(2_480_000.0, x_range[0]), min(2_840_000.0, x_range[1])
    ylo, yhi = max(1_070_000.0, y_range[0]), min(1_300_000.0, y_range[1])

    def _place(n: int) -> np.ndarray:
        return np.column_stack(
            [rng.uniform(xlo, xhi, n), rng.uniform(ylo, yhi, n)]
        )

    return {"obs": _place(22), "miss": _place(8), "query": _place(18)}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _plot_panel_a_with_data(
    ax: plt.Axes,
    t2m_nwp: np.ndarray,
    x_nwp: np.ndarray,
    y_nwp: np.ndarray,
) -> None:
    print("Loading topography …")
    dem, x_topo, y_topo = load_topo()

    # Upsample T_2M from coarse NWP grid to subsampled topo grid
    interp = RegularGridInterpolator(
        (y_nwp, x_nwp),
        t2m_nwp,
        method="linear",
        bounds_error=False,
        fill_value=float(np.nanmean(t2m_nwp)),
    )
    YY, XX = np.meshgrid(y_topo, x_topo, indexing="ij")  # (H', W')
    t2m_hires = interp(
        np.stack([YY.ravel(), XX.ravel()], axis=-1)
    ).reshape(YY.shape)

    # Standardise to FIELD_VLIM
    mu = float(np.nanmean(t2m_hires))
    sigma = float(np.nanstd(t2m_hires))
    t2m_norm = np.clip((t2m_hires - mu) / max(sigma, 1e-6) * 1.5, *FIELD_VLIM)

    H, W = dem.shape
    vignette = _make_vignette(H, W)
    ext = [x_topo[0], x_topo[-1], y_topo[0], y_topo[-1]]

    # ── Hillshaded terrain
    ls = LightSource(azdeg=310, altdeg=40)
    terrain_rgb = ls.shade(dem, cmap=TERRAIN_CMAP, vert_exag=1.8, blend_mode="soft")
    terrain_rgb[..., :3] = terrain_rgb[..., :3] * 0.60 + 0.40  # desaturate toward white
    ax.imshow(
        terrain_rgb,
        origin="lower",
        aspect="equal",
        interpolation="bilinear",
        extent=ext,
    )

    # ── NWP overlay with vignette fade
    nwp_norm = plt.Normalize(vmin=FIELD_VLIM[0], vmax=FIELD_VLIM[1])
    nwp_rgba = FIELD_CMAP(nwp_norm(t2m_norm)).copy()
    nwp_rgba[:, :, 3] = vignette * 0.68
    ax.imshow(
        nwp_rgba,
        origin="lower",
        aspect="equal",
        interpolation="bilinear",
        extent=ext,
    )

    # ── Invented stations
    stations = _invented_stations(
        (float(x_topo[0]), float(x_topo[-1])),
        (float(y_topo[0]), float(y_topo[-1])),
    )
    ax.scatter(
        stations["obs"][:, 0],
        stations["obs"][:, 1],
        c=ORANGE,
        s=80,
        zorder=6,
        edgecolors="white",
        linewidths=0.8,
        label="Observed station",
    )
    ax.scatter(
        stations["miss"][:, 0],
        stations["miss"][:, 1],
        c=GRAY_STA,
        s=55,
        zorder=6,
        edgecolors="white",
        linewidths=0.8,
        alpha=0.85,
        label="Missing observation",
    )
    ax.scatter(
        stations["query"][:, 0],
        stations["query"][:, 1],
        c=BLUE_Q,
        s=60,
        zorder=6,
        marker="D",
        edgecolors="white",
        linewidths=0.8,
        label="Query location",
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.legend(
        loc="lower left",
        fontsize=7,
        framealpha=0.92,
        markerscale=0.9,
        edgecolor="#CCCCCC",
    )
    ax.set_title("Inputs at time $t$  (real COSMO data)", fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=CMAP_FIELD, norm=nwp_norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.036, pad=0.02, label="T_2M [norm.]")


def _add_scale_bar(
    ax: plt.Axes, xhi: float, ylo: float, n_km: int = 5, bar_h: float = 380.0
) -> None:
    """Alternating black/white scale bar, one segment per km."""
    km = 1_000.0
    x_end = xhi - 2_000.0
    x_start = x_end - n_km * km
    y_bar = ylo + 1_400.0

    # Translucent white background — tall enough to cover the labels below the bar
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


# ---------------------------------------------------------------------------
# Zoom helpers
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


def load_t2m_zoom(
    t2m_nwp: np.ndarray, x_nwp: np.ndarray, y_nwp: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop pre-loaded NWP T_2M to zoom window; return native-resolution slice."""
    xlo, xhi, ylo, yhi = _zoom_bounds()
    xi0, xi1 = int(np.searchsorted(x_nwp, xlo)), int(np.searchsorted(x_nwp, xhi)) + 1
    yi0, yi1 = int(np.searchsorted(y_nwp, ylo)), int(np.searchsorted(y_nwp, yhi)) + 1
    xi0, xi1 = max(0, xi0), min(len(x_nwp), xi1)
    yi0, yi1 = max(0, yi0), min(len(y_nwp), yi1)
    return t2m_nwp[yi0:yi1, xi0:xi1], x_nwp[xi0:xi1], y_nwp[yi0:yi1]


def load_stations_zoom() -> np.ndarray:
    """Real station coordinates (LV95) clipped to the zoom window."""
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
    return tbl.loc[mask, ["swiss_easting", "swiss_northing"]].values


def plot_panel_a_zoom(
    ax: plt.Axes,
    t2m_nwp: np.ndarray,
    x_nwp: np.ndarray,
    y_nwp: np.ndarray,
) -> None:
    """Panel A zoomed to 50×50 km — high-res topo + blocky NWP overlay."""
    print("Loading topo for zoom …")
    dem, x_topo, y_topo = load_topo_zoom()

    t2m_crop, x_nwp_z, y_nwp_z = load_t2m_zoom(t2m_nwp, x_nwp, y_nwp)

    ref_field = t2m_crop if ZOOM_LOCAL_NORM else t2m_nwp
    mu = float(np.nanmean(ref_field))
    sigma = float(np.nanstd(ref_field))
    t2m_norm = np.clip((t2m_crop - mu) / max(sigma, 1e-6) * ZOOM_CONTRAST, *FIELD_VLIM)

    xlo, xhi, ylo, yhi = _zoom_bounds()
    ext_topo = [x_topo[0], x_topo[-1], y_topo[0], y_topo[-1]]

    # Cell boundaries: imshow extent expects edges, not centres
    dx_nwp = float(x_nwp_z[1] - x_nwp_z[0])
    dy_nwp = float(y_nwp_z[1] - y_nwp_z[0])
    bounds_x = np.concatenate([[x_nwp_z[0] - dx_nwp / 2], x_nwp_z + dx_nwp / 2])
    bounds_y = np.concatenate([[y_nwp_z[0] - dy_nwp / 2], y_nwp_z + dy_nwp / 2])
    ext_nwp = [bounds_x[0], bounds_x[-1], bounds_y[0], bounds_y[-1]]

    # ── Hillshaded terrain at 100 m resolution
    ls = LightSource(azdeg=310, altdeg=40)
    terrain_rgb = ls.shade(dem, cmap=TERRAIN_CMAP, vert_exag=2.5, blend_mode="soft")
    terrain_rgb[..., :3] = terrain_rgb[..., :3] * 0.60 + 0.40  # desaturate toward white
    ax.imshow(
        terrain_rgb,
        origin="lower",
        aspect="equal",
        interpolation="bilinear",
        extent=ext_topo,
    )

    # ── NWP overlay — blocky pixels + vignette fade at borders
    nwp_norm_obj = plt.Normalize(vmin=FIELD_VLIM[0], vmax=FIELD_VLIM[1])
    nwp_rgba = FIELD_CMAP(nwp_norm_obj(t2m_norm)).copy()
    nwp_rgba[:, :, 3] = _make_vignette(*t2m_norm.shape, edge_frac=0.30) * 0.72
    ax.imshow(
        nwp_rgba,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        extent=ext_nwp,
    )

    # ── NWP grid lines at cell boundaries
    for xv in bounds_x:
        ax.axvline(xv, color="white", lw=0.3, alpha=0.4, zorder=3)
    for yv in bounds_y:
        ax.axhline(yv, color="white", lw=0.3, alpha=0.4, zorder=3)

    # ── Station locations — canonical split shared across all panels
    print("Loading real stations …")
    obs_sta, miss_sta = canonical_stations()

    ax.scatter(
        obs_sta[:, 0], obs_sta[:, 1],
        c=ORANGE, s=80, zorder=6, edgecolors="white", linewidths=0.8,
        label="Observed station",
    )
    ax.scatter(
        miss_sta[:, 0], miss_sta[:, 1],
        c=GRAY_STA, s=55, zorder=6, edgecolors="white", linewidths=0.8,
        alpha=0.85, label="Missing observation",
    )

    # ── Query locations — canonical positions shared across all panels
    queries = canonical_queries()
    ax.scatter(
        queries[:, 0], queries[:, 1],
        c=BLUE_Q, s=60, zorder=6, marker="D", edgecolors="white", linewidths=0.8,
        label="Query location",
    )
    grid_queries = canonical_queries_grid()
    ax.scatter(
        grid_queries[:, 0], grid_queries[:, 1],
        c=BLUE_Q, s=60, zorder=6, marker="D", edgecolors="white", linewidths=0.8,
    )

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


    # ── Scale bar (5 km, alternating black/white per km)
    _add_scale_bar(ax, xhi, ylo)

    # ── Resolution patches embedded as legend handles
    # NWP: 4×4 crop from centre — keeps blocky pixel texture clearly visible
    ch, cw = t2m_norm.shape[0] // 2, t2m_norm.shape[1] // 2
    nwp_patch_rgba = FIELD_CMAP(nwp_norm_obj(t2m_norm[ch - 2 : ch + 2, cw - 2 : cw + 2]))
    # Topo: 40×40 crop from centre at 100 m — same 4 km area, fine detail
    th, tw = terrain_rgb.shape[0] // 2, terrain_rgb.shape[1] // 2
    topo_patch = terrain_rgb[th - 20 : th + 20, tw - 20 : tw + 20]

    nwp_proxy = mpatches.Patch()
    topo_proxy = mpatches.Patch()
    handles, labels = ax.get_legend_handles_labels()
    handles += [nwp_proxy, topo_proxy]
    labels += ["NWP (low res.)", "Topo. (high res.)"]

    leg = ax.legend(
        handles, labels,
        handler_map={
            nwp_proxy: _PatchImageHandler(nwp_patch_rgba, interp="nearest", yoffset=0.9),
            topo_proxy: _PatchImageHandler(topo_patch, interp="bilinear", yoffset=0.9),
        },
        loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
        fontsize=12, framealpha=0.92,
        handleheight=2.2, handlelength=2.2,
        markerscale=1.4, edgecolor="#CCCCCC",
        labelspacing=0.6, columnspacing=1.6, borderpad=0.8,
    )
    leg.set_zorder(10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load NWP once; reuse for both panels
    print("Loading T_2M from NWP …")
    t2m_nwp, x_nwp, y_nwp = load_t2m_on_ref_grid()

    # ── Full domain panel
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    _plot_panel_a_with_data(ax, t2m_nwp, x_nwp, y_nwp)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PATH.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_PATH.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")

    # ── Zoom panel
    zoom_path = OUT_PATH.parent / "oat_panel_a_zoom.png"
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    plot_panel_a_zoom(ax, t2m_nwp, x_nwp, y_nwp)
    fig.tight_layout()
    fig.savefig(zoom_path, dpi=200, bbox_inches="tight")
    fig.savefig(zoom_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(zoom_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {zoom_path}")


if __name__ == "__main__":
    main()
