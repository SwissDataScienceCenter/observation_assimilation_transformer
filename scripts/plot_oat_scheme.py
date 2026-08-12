"""Generate OAT scheme figure panels.

Run: poetry run python -m scripts.plot_oat_scheme
Outputs: data/figures/oat_scheme_panel_*.png  +  oat_scheme_generated.pdf
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, LightSource
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.ndimage import gaussian_filter
from scipy.stats import logistic, norm

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
ORANGE = "#E8732A"
ORANGE_L = "#F5C080"
GRAY_STA = "#AAAAAA"
BLUE_Q = "#3A72C6"
BLUE_L = "#A0BEEC"
GREEN = "#2BA02B"
DARK_BLUE = "#1A3A6B"
CMAP_FIELD = "RdBu_r"
CMAP_UNC = "YlOrBr"

# Terrain colormap: valleys → forest → highland → rock → snow
TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    "terrain_oat",
    [
        (0.00, "#3B5E1F"),  # dark forest green
        (0.12, "#5C8A3A"),  # medium green
        (0.28, "#8EAA62"),  # light green
        (0.42, "#C8B57A"),  # sandy/meadow
        (0.55, "#96703C"),  # warm brown (treeline)
        (0.68, "#7A6050"),  # dark rock brown
        (0.80, "#A09898"),  # gray rock
        (0.90, "#D2D2D2"),  # light gray
        (1.00, "#F4F4F4"),  # near-white snow
    ],
)

SEED = 42
GS = 64
FIELD_VLIM = (-3.5, 3.5)
UNC_VLIM = (0.0, 2.5)

OUT_DIR = Path(__file__).parents[1] / "data" / "figures"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)


# ---------------------------------------------------------------------------
# Topography & vignette helpers
# ---------------------------------------------------------------------------
def _generate_topo(rng: np.random.RandomState, gs: int = GS) -> np.ndarray:
    """Synthetic Alpine-like elevation field (metres, roughly 0–4000)."""
    XX, YY = np.meshgrid(np.linspace(0, 1, gs), np.linspace(0, 1, gs))

    # Main Alpine arc sweeping diagonally across centre
    alpine = (
        2400 * np.exp(-((XX - 0.48 + 0.18 * (YY - 0.5)) ** 2 / 0.11 + (YY - 0.44) ** 2 / 0.16))
        + 1900 * np.exp(-((XX - 0.32 + 0.1 * (YY - 0.5)) ** 2 / 0.055 + (YY - 0.50) ** 2 / 0.07))
        + 1700 * np.exp(-((XX - 0.66 + 0.1 * (YY - 0.5)) ** 2 / 0.050 + (YY - 0.41) ** 2 / 0.06))
    )
    # Pre-Alps / Jura (north, lower)
    pre_alps = 750 * np.exp(-((XX - 0.38) ** 2 / 0.07 + (YY - 0.73) ** 2 / 0.055))
    jura = 620 * np.exp(-((XX - 0.21) ** 2 / 0.035 + (YY - 0.79) ** 2 / 0.045))
    # Southern valleys / Po plain (south-east edge)
    valleys = -400 * np.exp(-((XX - 0.75) ** 2 / 0.10 + (YY - 0.22) ** 2 / 0.10))

    # Multi-scale noise for realistic texture
    noise = (
        gaussian_filter(rng.randn(gs, gs), sigma=14) * 480
        + gaussian_filter(rng.randn(gs, gs), sigma=5) * 200
        + gaussian_filter(rng.randn(gs, gs), sigma=1.5) * 70
    )

    topo = alpine + pre_alps + jura + valleys + noise + 280
    return np.clip(topo, 0, 4500)


def _make_vignette(gs: int = GS, edge_frac: float = 0.26) -> np.ndarray:
    """Smoothstep vignette: 1 at centre, 0 at all four edges."""
    u = np.linspace(0, 1, gs)
    XX, YY = np.meshgrid(u, u)

    def _step(v: np.ndarray) -> np.ndarray:
        t = np.clip(v / edge_frac, 0, 1)
        return t * t * (3 - 2 * t)  # smoothstep

    return _step(XX) * _step(1 - XX) * _step(YY) * _step(1 - YY)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
def generate_data() -> dict:
    rng = np.random.RandomState(SEED)
    xx = np.arange(GS, dtype=float)
    yy = np.arange(GS, dtype=float)
    XX, YY = np.meshgrid(xx, yy)

    field_nwp = (
        gaussian_filter(rng.randn(GS, GS), sigma=18) * 3.0
        + gaussian_filter(rng.randn(GS, GS), sigma=7) * 1.5
        + gaussian_filter(rng.randn(GS, GS), sigma=2.5) * 0.4
    )
    field_prior = gaussian_filter(field_nwp, sigma=4)

    def _place(n: int, margin: int = 5) -> np.ndarray:
        return np.column_stack(
            [rng.uniform(margin, GS - margin, n), rng.uniform(margin, GS - margin, n)]
        )

    stations_obs = _place(22)
    stations_miss = _place(8)
    queries = _place(18)
    innovations = rng.randn(22) * 0.9

    field_post = field_prior.copy()
    for (xi, yi), inn in zip(stations_obs, innovations):
        dist2 = (XX - xi) ** 2 + (YY - yi) ** 2
        field_post += inn * np.exp(-dist2 / (2 * 5.5**2))

    uncertainty = np.full((GS, GS), 2.2)
    for xi, yi in stations_obs:
        dist2 = (XX - xi) ** 2 + (YY - yi) ** 2
        uncertainty -= 1.9 * np.exp(-dist2 / (2 * 7.0**2))
    uncertainty = np.clip(uncertainty, 0.05, 2.5)

    topo = _generate_topo(rng)
    vignette = _make_vignette(GS)

    return dict(
        XX=XX,
        YY=YY,
        field_nwp=field_nwp,
        field_prior=field_prior,
        field_post=field_post,
        uncertainty=uncertainty,
        stations_obs=stations_obs,
        stations_miss=stations_miss,
        queries=queries,
        innovations=innovations,
        topo=topo,
        vignette=vignette,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _knn_edges(pos: np.ndarray, k: int) -> list[tuple[int, int]]:
    edges = []
    for i in range(len(pos)):
        dists = np.linalg.norm(pos - pos[i], axis=1)
        dists[i] = np.inf
        for j in np.argsort(dists)[:k]:
            edges.append((i, int(j)))
    return edges


def _decay_weight(pi: np.ndarray, pj: np.ndarray, sigma: float = 0.35) -> float:
    d = float(np.linalg.norm(pi - pj))
    return float(np.exp(-(d**2) / (2 * sigma**2)))


def _norm01(pos: np.ndarray) -> np.ndarray:
    mn, mx = pos.min(0), pos.max(0)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    return (pos - mn) / rng


def _colorbar(ax: plt.Axes, im, label: str) -> None:
    plt.colorbar(im, ax=ax, fraction=0.038, pad=0.025, label=label)


# ---------------------------------------------------------------------------
# Panel A — Inputs
# ---------------------------------------------------------------------------
def plot_panel_A(ax: plt.Axes, data: dict) -> None:
    topo = data["topo"]
    vignette = data["vignette"]

    # ── 1. Hillshaded terrain background ────────────────────────────────────
    ls = LightSource(azdeg=310, altdeg=40)
    terrain_rgb = ls.shade(
        topo,
        cmap=TERRAIN_CMAP,
        vert_exag=1.8,
        blend_mode="soft",
    )
    ax.imshow(terrain_rgb, origin="lower", aspect="equal", interpolation="bilinear")

    # ── 2. NWP field overlay — fades to transparent at borders ──────────────
    nwp_norm = plt.Normalize(vmin=FIELD_VLIM[0], vmax=FIELD_VLIM[1])
    nwp_rgba = plt.cm.RdBu_r(nwp_norm(data["field_nwp"])).copy()
    nwp_rgba[:, :, 3] = vignette * 0.68   # centre ≈ 68 % opaque, edges → 0
    ax.imshow(nwp_rgba, origin="lower", aspect="equal", interpolation="bilinear")

    # ── 3. Station / query markers ───────────────────────────────────────────
    ax.scatter(
        data["stations_obs"][:, 0], data["stations_obs"][:, 1],
        c=ORANGE, s=80, zorder=6, edgecolors="white", linewidths=0.8,
        label="Observed station",
    )
    ax.scatter(
        data["stations_miss"][:, 0], data["stations_miss"][:, 1],
        c=GRAY_STA, s=55, zorder=6, edgecolors="white", linewidths=0.8,
        alpha=0.85, label="Missing observation",
    )
    ax.scatter(
        data["queries"][:, 0], data["queries"][:, 1],
        c=BLUE_Q, s=60, zorder=6, marker="D", edgecolors="white", linewidths=0.8,
        label="Query location",
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.92, markerscale=0.9,
              edgecolor="#CCCCCC")
    ax.set_title("Inputs at time $t$", fontsize=9)

    # NWP colorbar (slim, on the right)
    sm = plt.cm.ScalarMappable(cmap=CMAP_FIELD, norm=nwp_norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.036, pad=0.02, label="NWP field [norm.]")


# ---------------------------------------------------------------------------
# Panel B.1 — Prior distribution
# ---------------------------------------------------------------------------
def plot_panel_B_dist(ax: plt.Axes) -> None:
    x = np.linspace(-4, 4, 300)
    pdf = norm.pdf(x, 0, 1)
    ax.plot(x, pdf, color=DARK_BLUE, lw=2.0)
    ax.fill_between(x, pdf, where=(np.abs(x) <= 1), alpha=0.22, color=BLUE_Q)
    ax.fill_between(x, pdf, where=(np.abs(x) <= 2), alpha=0.10, color=BLUE_Q)
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, None)
    ax.set_xlabel("$y$", fontsize=9)
    ax.set_title(r"$p_{\mathrm{prior}}(y)$   (example variable)", fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


# ---------------------------------------------------------------------------
# Panel B.2 — Prior spatial map
# ---------------------------------------------------------------------------
def plot_panel_B_map(ax: plt.Axes, data: dict) -> None:
    im = ax.imshow(
        data["field_prior"],
        cmap=CMAP_FIELD,
        vmin=FIELD_VLIM[0],
        vmax=FIELD_VLIM[1],
        origin="lower",
        aspect="auto",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Prior mean (example variable)", fontsize=9)
    _colorbar(ax, im, "Temp. [norm.]")


# ---------------------------------------------------------------------------
# Panel C.1 — Innovation CDFs
# ---------------------------------------------------------------------------
def plot_panel_C_innovation(ax: plt.Axes) -> None:
    colors = [BLUE_Q, ORANGE, GREEN]
    lambdas = [1.8, 0.1, -1.5]
    # Different prior shapes for each variable
    params = [
        {"loc": 5.5, "scale": 1.1},
        {"loc": 5.5, "scale": 1.7},
        {"loc": 5.5, "scale": 0.9},
    ]
    x_range = np.linspace(0.5, 11.5, 300)

    for lam, p, color in zip(lambdas, params, colors):
        cdf_vals = logistic.cdf(x_range, loc=p["loc"], scale=p["scale"])
        cdf_target = norm.cdf(lam)
        x_obs = float(logistic.ppf(cdf_target, loc=p["loc"], scale=p["scale"]))

        ax.plot(x_range, cdf_vals, color=color, lw=2.0)
        ax.axvline(x_obs, color=color, linestyle="--", lw=1.2, alpha=0.7)
        ax.axhline(cdf_target, color=color, linestyle=":", lw=1.0, alpha=0.5)
        ax.scatter([x_obs], [cdf_target], color=color, s=40, zorder=6)

        sign = "+" if lam >= 0 else ""
        ax.annotate(
            f"$\\lambda = {sign}{lam:.1f}$",
            xy=(11.6, cdf_target),
            fontsize=8,
            color=color,
            va="center",
            fontweight="bold",
        )

    ax.set_xlim(0.5, 14.5)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("Observation $y$", fontsize=9)
    ax.set_ylabel("Prior CDF", fontsize=9)
    ax.set_title("Innovation per variable", fontsize=9)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Formula annotation — upper-left to avoid overlap with λ = -1.5 at bottom-right
    ax.text(
        0.03,
        0.97,
        r"$h = \Phi^{-1}(\hat{F}(y))$",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=8.5,
        style="italic",
        color="#444444",
    )


# ---------------------------------------------------------------------------
# Panel C.2 — Fusion gate block diagram
# ---------------------------------------------------------------------------
def plot_panel_C_gate(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.2, 8.5)
    ax.axis("off")

    def _box(
        cx: float,
        cy: float,
        w: float,
        h: float,
        label: str,
        fc: str = "#D8E8FF",
        fs: int = 9,
    ) -> None:
        rect = FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.12",
            facecolor=fc,
            edgecolor="#777777",
            linewidth=1.0,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fs)

    def _arr(
        x1: float, y1: float, x2: float, y2: float, label: str = "", fc: str = "#555555"
    ) -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=fc, lw=1.2),
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(
                mx + 0.15, my, label, fontsize=7.5, color="#555555", va="center"
            )

    # ─ inputs ─
    _box(2.5, 7.8, 2.2, 0.9, "$\\mathbf{x}$\n(prior token)", fc="#D5E8FF")
    _box(7.5, 7.8, 2.6, 0.9, "$\\mathbf{x}_y$\n(innov. token)", fc="#FFE5CC")

    # ─ gate ─
    _box(5.0, 5.5, 3.0, 1.0, "$g(\\mathbf{m}^s)$  (gate)", fc=ORANGE_L)
    _box(9.2, 5.5, 1.4, 0.8, "$\\mathbf{m}^s$", fc="#EEEEEE", fs=8)

    # ─ multiply symbols ─
    ax.text(2.5, 3.5, "$\\times$", ha="center", va="center", fontsize=15, color="#444")
    ax.text(7.5, 3.5, "$\\times$", ha="center", va="center", fontsize=15, color="#444")

    # ─ sum ─
    circ = plt.Circle((5.0, 1.9), 0.38, fc="white", ec="#777777", lw=1.0, zorder=3)
    ax.add_patch(circ)
    ax.text(5.0, 1.9, "$\\oplus$", ha="center", va="center", fontsize=13, zorder=4)

    # ─ output ─
    _box(5.0, 0.4, 2.2, 0.8, "$\\mathbf{z}_y$  (output token)", fc="#D5FFD5")

    # ─ arrows ─
    _arr(2.5, 7.35, 2.5, 3.82)             # x → ×_left
    _arr(7.5, 7.35, 7.5, 3.82)             # x_y → ×_right
    _arr(9.2, 5.1, 6.52, 5.5)              # mask → gate
    _arr(3.95, 5.5, 2.75, 3.82)            # gate (1-g) → ×_left
    _arr(6.05, 5.5, 7.25, 3.82)            # gate (g) → ×_right
    _arr(2.5, 3.18, 4.62, 2.2)             # ×_left → ⊕
    _arr(7.5, 3.18, 5.38, 2.2)             # ×_right → ⊕
    _arr(5.0, 1.52, 5.0, 0.8)              # ⊕ → z_y

    # ─ gate labels ─
    ax.text(3.2, 4.6, "$1{-}g$", fontsize=7.5, color="#666")
    ax.text(6.5, 4.6, "$g$", fontsize=7.5, color="#666")

    # ─ formula ─
    ax.text(
        5.0,
        -0.75,
        r"$\mathbf{z}_y = (1 - g(\mathbf{m}^s))\odot\mathbf{x}$"
        r"$ + g(\mathbf{m}^s)\odot\mathbf{x}_y$",
        ha="center",
        va="top",
        fontsize=8.5,
        style="italic",
    )
    ax.set_title("Fusion with gating", fontsize=9)


# ---------------------------------------------------------------------------
# Panel D.1 — Station self-attention
# ---------------------------------------------------------------------------
def plot_panel_D_self_attn(ax: plt.Axes, data: dict) -> None:
    pos = _norm01(data["stations_obs"])
    edges = _knn_edges(pos, k=4)

    for i, j in edges:
        w = _decay_weight(pos[i], pos[j], sigma=0.38)
        ax.plot(
            [pos[i, 0], pos[j, 0]],
            [pos[i, 1], pos[j, 1]],
            color=ORANGE_L,
            lw=0.8 + w * 2.2,
            alpha=0.15 + w * 0.65,
            zorder=2,
        )

    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=ORANGE,
        s=75,
        zorder=4,
        edgecolors="white",
        linewidths=0.7,
    )
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 1.06)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Station (K/V) self-attention\n(spatial mixing)", fontsize=9)


# ---------------------------------------------------------------------------
# Panel D.2 — Cross-attention (stations → queries)
# ---------------------------------------------------------------------------
def plot_panel_D_cross_attn(ax: plt.Axes, data: dict) -> None:
    all_pos = np.vstack([data["stations_obs"], data["queries"]])
    mn, mx = all_pos.min(0), all_pos.max(0)
    rng_ = np.where(mx - mn == 0, 1.0, mx - mn)
    pos_s = (data["stations_obs"] - mn) / rng_
    pos_q = (data["queries"] - mn) / rng_

    k = 3
    for qi in range(len(pos_q)):
        dists = np.linalg.norm(pos_s - pos_q[qi], axis=1)
        for si in np.argsort(dists)[:k]:
            w = _decay_weight(pos_q[qi], pos_s[si], sigma=0.42)
            ax.plot(
                [pos_q[qi, 0], pos_s[si, 0]],
                [pos_q[qi, 1], pos_s[si, 1]],
                color=BLUE_L,
                lw=0.6 + w * 1.8,
                alpha=0.12 + w * 0.55,
                zorder=2,
            )

    ax.scatter(
        pos_s[:, 0],
        pos_s[:, 1],
        c=ORANGE,
        s=65,
        zorder=4,
        edgecolors="white",
        linewidths=0.7,
        label="Station (K, V)",
    )
    ax.scatter(
        pos_q[:, 0],
        pos_q[:, 1],
        c=BLUE_Q,
        s=50,
        zorder=5,
        marker="D",
        edgecolors="white",
        linewidths=0.7,
        label="Query (Q)",
    )
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 1.06)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=6.5, framealpha=0.85)
    ax.set_title("Cross-attention to query locations\n(propagate information)", fontsize=9)


# ---------------------------------------------------------------------------
# Panel D.3 — Geometry-aware attention bias
# ---------------------------------------------------------------------------
def plot_panel_D_geo_bias(ax: plt.Axes) -> None:
    d = np.linspace(0, 3.5, 300)
    beta = 1.55 * np.exp(-1.9 * d)

    ax.plot(d, beta, color=DARK_BLUE, lw=2.2)
    ax.fill_between(d, 0, beta, alpha=0.13, color=DARK_BLUE)
    ax.axhline(0, color="gray", lw=0.7, linestyle="--", alpha=0.55)

    ax.set_xlim(0, 3.5)
    ax.set_ylim(-0.12, 1.85)
    ax.set_xlabel("Distance $d$", fontsize=9)
    ax.set_ylabel(r"$\beta(d)$", fontsize=9)
    ax.set_title("Geometry-aware attention bias", fontsize=9)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.text(
        0.97,
        0.97,
        r"$\mathbf{Q}_i^\top \mathbf{K}_j + \beta(\|\mathbf{x}_i - \mathbf{x}_j\|)$",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
        style="italic",
        color=DARK_BLUE,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#CCCCCC"
        ),
    )


# ---------------------------------------------------------------------------
# Panel E.1 — Posterior mean
# ---------------------------------------------------------------------------
def plot_panel_E_mean(ax: plt.Axes, data: dict) -> None:
    im = ax.imshow(
        data["field_post"],
        cmap=CMAP_FIELD,
        vmin=FIELD_VLIM[0],
        vmax=FIELD_VLIM[1],
        origin="lower",
        aspect="auto",
    )
    ax.scatter(
        data["stations_obs"][:, 0],
        data["stations_obs"][:, 1],
        c=ORANGE,
        s=50,
        zorder=5,
        edgecolors="white",
        linewidths=0.6,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Posterior mean (with observations)", fontsize=9)
    _colorbar(ax, im, "Temp. [norm.]")


# ---------------------------------------------------------------------------
# Panel E.2 — Uncertainty
# ---------------------------------------------------------------------------
def plot_panel_E_std(ax: plt.Axes, data: dict) -> None:
    im = ax.imshow(
        data["uncertainty"],
        cmap=CMAP_UNC,
        vmin=UNC_VLIM[0],
        vmax=UNC_VLIM[1],
        origin="lower",
        aspect="auto",
    )
    ax.scatter(
        data["stations_obs"][:, 0],
        data["stations_obs"][:, 1],
        c="white",
        s=35,
        zorder=5,
        marker="o",
        edgecolors=ORANGE,
        linewidths=1.0,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Predictive uncertainty", fontsize=9)
    _colorbar(ax, im, "Std. [norm.]")


# ---------------------------------------------------------------------------
# Combined figure
# ---------------------------------------------------------------------------
_SECTION_LABELS = [
    ("A", "Inputs"),
    ("B", "Prior (without obs.)"),
    ("C", "Innovation & Assimilation"),
    ("D", "OAT"),
    ("E", "Posterior (with obs.)"),
]


def assemble_figure(data: dict) -> plt.Figure:
    fig = plt.figure(figsize=(22, 11), facecolor="white")

    # Outer 5-column GridSpec — leave top margin for section headers
    gs = GridSpec(
        2,
        5,
        figure=fig,
        hspace=0.48,
        wspace=0.40,
        left=0.03,
        right=0.97,
        top=0.89,
        bottom=0.06,
    )

    # ── A: full-height ──────────────────────────────────────────────────────
    ax_A = fig.add_subplot(gs[:, 0])
    plot_panel_A(ax_A, data)

    # ── B: distribution (top) + spatial map (bottom) ────────────────────────
    ax_B1 = fig.add_subplot(gs[0, 1])
    plot_panel_B_dist(ax_B1)

    ax_B2 = fig.add_subplot(gs[1, 1])
    plot_panel_B_map(ax_B2, data)

    # ── C: innovation CDFs (top) + gating (bottom) ──────────────────────────
    ax_C1 = fig.add_subplot(gs[0, 2])
    plot_panel_C_innovation(ax_C1)

    ax_C2 = fig.add_subplot(gs[1, 2])
    plot_panel_C_gate(ax_C2)

    # ── D: three rows in column 3 ───────────────────────────────────────────
    gs_D = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[:, 3], hspace=0.65)
    ax_D1 = fig.add_subplot(gs_D[0])
    plot_panel_D_self_attn(ax_D1, data)

    ax_D2 = fig.add_subplot(gs_D[1])
    plot_panel_D_cross_attn(ax_D2, data)

    ax_D3 = fig.add_subplot(gs_D[2])
    plot_panel_D_geo_bias(ax_D3)

    # ── E: posterior mean (top) + uncertainty (bottom) ──────────────────────
    ax_E1 = fig.add_subplot(gs[0, 4])
    plot_panel_E_mean(ax_E1, data)

    ax_E2 = fig.add_subplot(gs[1, 4])
    plot_panel_E_std(ax_E2, data)

    # ── Section header labels — placed above each column ────────────────────
    # Collect the x-centre of each column's leftmost/only axes
    col_axes = [ax_A, ax_B1, ax_C1, ax_D1, ax_E1]
    for ax_ref, (letter, desc) in zip(col_axes, _SECTION_LABELS):
        # Get the axes bounding box in figure coordinates
        pos = ax_ref.get_position()
        x_center = pos.x0  # left-align label with the left edge
        y_top = pos.y1 + 0.015
        fig.text(
            x_center,
            y_top,
            f"$\\mathbf{{{letter}}}$   {desc}",
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#1A1A1A",
            fontweight="semibold",
        )

    return fig


# ---------------------------------------------------------------------------
# Individual panels
# ---------------------------------------------------------------------------
def _panel_specs(data: dict) -> list[tuple[str, float, float, object]]:
    """(label, width, height, draw_fn)"""
    return [
        ("A_inputs", 5.0, 5.0, lambda ax: plot_panel_A(ax, data)),
        ("B1_prior_dist", 3.5, 3.0, lambda ax: plot_panel_B_dist(ax)),
        ("B2_prior_map", 3.5, 3.5, lambda ax: plot_panel_B_map(ax, data)),
        ("C1_innovation", 4.0, 3.5, lambda ax: plot_panel_C_innovation(ax)),
        ("C2_gate", 4.0, 4.5, lambda ax: plot_panel_C_gate(ax)),
        ("D1_self_attn", 3.5, 3.0, lambda ax: plot_panel_D_self_attn(ax, data)),
        ("D2_cross_attn", 3.5, 3.0, lambda ax: plot_panel_D_cross_attn(ax, data)),
        ("D3_geo_bias", 3.5, 2.8, lambda ax: plot_panel_D_geo_bias(ax)),
        ("E1_posterior", 3.5, 3.5, lambda ax: plot_panel_E_mean(ax, data)),
        ("E2_uncertainty", 3.5, 3.5, lambda ax: plot_panel_E_std(ax, data)),
    ]


def save_individual_panels(data: dict) -> None:
    for label, w, h, fn in _panel_specs(data):
        fig, ax = plt.subplots(figsize=(w, h))
        fn(ax)
        fig.tight_layout()
        path = OUT_DIR / f"oat_scheme_panel_{label}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating synthetic data …")
    data = generate_data()

    print("Assembling combined figure …")
    fig = assemble_figure(data)
    for suffix in (".pdf", ".png"):
        path = OUT_DIR / f"oat_scheme_generated{suffix}"
        fig.savefig(path, dpi=200 if suffix == ".png" else None, bbox_inches="tight")
        print(f"  {path.name}")
    plt.close(fig)

    print("Saving individual panels …")
    save_individual_panels(data)
    print("Done.")


if __name__ == "__main__":
    main()
