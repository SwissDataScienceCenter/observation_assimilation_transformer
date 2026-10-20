#!/usr/bin/env bash
# Figure 1 (fig:oat): the data panel (left half) of the OAT scheme.
# The architecture diagram (right half) was drawn by hand.
# Needs: the Swiss data (the crop to Switzerland is enough).
# The panel uses the zoom image of the panel script, not the full-domain image.
# Output: figures/out/fig1_a.pdf
source "$(dirname "$0")/_lib.sh"

need_swiss
run python -m scripts.plots.oat_scheme.plot_oat_panel_a_real
run cp data/figures/oat_panel_a_zoom.pdf "$OUT_DIR/fig1_a.pdf"
