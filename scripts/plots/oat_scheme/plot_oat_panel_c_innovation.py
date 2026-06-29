"""Panel C (innovation) of the OAT scheme.

Shows how normalised station observations are mapped to innovations through
the composed 1-D function  h = Φ⁻¹(F̂(y)):

  obs y  ─[Prior CDF  F̂]──►  p ∈ (0,1)  ─[Probit  Φ⁻¹]──►  h

Run: poetry run python -m scripts.plot_oat_panel_c_innovation
Output: data/figures/oat_panel_c_innovation.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import logistic, norm

# ---------------------------------------------------------------------------
# Palette (same as other panel scripts)
# ---------------------------------------------------------------------------
ORANGE = "#E8732A"
BLUE_Q = "#3A72C6"
GREEN = "#2BA02B"
DARK_BLUE = "#1A3A6B"
BLUE_L = "#A0BEEC"

OUT_PATH = Path(__file__).parents[3] / "data/figures/oat_panel_c_innovation.png"

plt.rcParams.update(
    {"font.family": "sans-serif", "font.size": 9, "axes.linewidth": 0.8}
)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
def plot_panel_c_innovation(ax: plt.Axes) -> None:
    """Two-step composition diagram: Prior CDF → Probit → innovation."""
    ax.axis("off")

    # Two inset axes side by side — leave extra room at the bottom for h labels
    ax_cdf = ax.inset_axes([0.03, 0.28, 0.44, 0.58])
    ax_prob = ax.inset_axes([0.54, 0.28, 0.44, 0.58])

    from matplotlib.patches import ConnectionPatch

    fig = ax.get_figure()

    # Three example observations shown in distinct colors
    colors = [BLUE_Q, ORANGE, GREEN]
    obs_vals = [7.2, 5.5, 3.2]          # example observations (normalised space)
    prior_loc, prior_scale = 5.4, 1.25  # single logistic prior

    x_range = np.linspace(1.5, 9.5, 400)
    h_range = np.linspace(-2.8, 2.8, 400)
    YLIM = (0.0, 1.0)  # shared y-axis range (probability), clamped to [0, 1]

    # Pre-compute p and h for each example
    p_vals = [float(logistic.cdf(obs, loc=prior_loc, scale=prior_scale)) for obs in obs_vals]
    h_vals = [float(norm.ppf(p)) for p in p_vals]

    # ── Left panel: Prior CDF  F̂(y) ─────────────────────────────────────
    cdf_curve = logistic.cdf(x_range, loc=prior_loc, scale=prior_scale)
    ax_cdf.plot(x_range, cdf_curve, color=DARK_BLUE, lw=2.2, zorder=3)
    ax_cdf.fill_between(x_range, cdf_curve, alpha=0.09, color=BLUE_Q)

    for obs, p, color in zip(obs_vals, p_vals, colors):
        # Vertical dashed: x-axis up to curve dot
        ax_cdf.plot([obs, obs], [0.0, p], color=color, lw=1.0, ls="--", alpha=0.70, zorder=2)
        # Horizontal dashed: curve dot to right edge (continues into right panel)
        ax_cdf.plot([obs, x_range[-1]], [p, p], color=color, lw=1.0, ls="--", alpha=0.70, zorder=2)
        ax_cdf.scatter([obs], [p], color=color, s=52, zorder=6, edgecolors="white", linewidths=0.8)

    ax_cdf.set_xlabel("obs $y$ (norm.)", fontsize=8.5)
    ax_cdf.set_ylabel("probability $p$", fontsize=8.5)
    ax_cdf.set_title("Prior CDF  $\\hat{F}$", fontsize=9, pad=4)
    ax_cdf.set_xlim(x_range[0], x_range[-1])
    ax_cdf.set_ylim(*YLIM)
    ax_cdf.set_yticks([0, 0.5, 1])
    ax_cdf.spines["right"].set_visible(False)
    ax_cdf.spines["top"].set_visible(False)
    ax_cdf.tick_params(labelsize=7.5)
    ax_cdf.set_box_aspect(1)

    # ── Right panel: Gaussian CDF  Φ(h) — h on x-axis ───────────────────
    gauss_curve = norm.cdf(h_range)
    ax_gauss = ax_prob
    ax_gauss.plot(h_range, gauss_curve, color=DARK_BLUE, lw=2.2, zorder=3)
    ax_gauss.fill_between(h_range, gauss_curve, alpha=0.09, color=BLUE_Q)

    for h, p, color in zip(h_vals, p_vals, colors):
        # Horizontal dashed: enters from left edge at level p
        ax_gauss.plot([h_range[0], h], [p, p], color=color, lw=1.0, ls="--", alpha=0.70, zorder=2)
        # Vertical dashed: curve dot down to x-axis
        ax_gauss.plot([h, h], [0.0, p], color=color, lw=1.0, ls="--", alpha=0.70, zorder=2)
        ax_gauss.scatter([h], [p], color=color, s=52, zorder=6, edgecolors="white", linewidths=0.8)
        sign = "+" if h >= 0 else ""
        # Annotate value just to the right of the dot
        h_offset = 0.12
        ax_gauss.text(
            h + h_offset, p,
            f"${sign}{h:.1f}$",
            ha="left", va="center",
            color=color, fontsize=8, fontweight="bold",
        )

    ax_gauss.set_xlabel("innovation $h$", fontsize=8.5)
    ax_gauss.set_title(r"Gaussian CDF  $\Phi$", fontsize=9, pad=4)
    ax_gauss.set_xlim(h_range[0], h_range[-1] + 0.5)
    ax_gauss.set_ylim(*YLIM)
    ax_gauss.set_yticks([0, 0.5, 1])
    ax_gauss.set_yticklabels([])  # shared y-axis — labels on left panel only
    ax_gauss.spines["right"].set_visible(False)
    ax_gauss.spines["top"].set_visible(False)
    ax_gauss.tick_params(labelsize=7.5)
    ax_gauss.set_box_aspect(1)

    # ── Connecting lines spanning the gap between panels ─────────────────
    # ConnectionPatch links (x_range[-1], p) in ax_cdf data to (h_range[0], p)
    # in ax_gauss data — at the same probability level p, bridging the gap.
    for p, color in zip(p_vals, colors):
        con = ConnectionPatch(
            xyA=(x_range[-1], p),
            xyB=(h_range[0], p),
            coordsA="data",
            coordsB="data",
            axesA=ax_cdf,
            axesB=ax_gauss,
            color=color,
            lw=1.0,
            ls="--",
            alpha=0.70,
            clip_on=False,
            zorder=10,
            arrowstyle="-|>",
            mutation_scale=10,
        )
        fig.add_artist(con)

    # ── Composed formula centred below the two panels ────────────────────
    ax.text(
        0.5,
        0.08,
        r"$h = \Phi^{-1}\!\left(\hat{F}(y)\right)$",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=11,
        style="italic",
        color="#333333",
    )

    ax.set_title("Innovation (per variable)", fontsize=10, pad=8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor="white")
    plot_panel_c_innovation(ax)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PATH.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_PATH.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
