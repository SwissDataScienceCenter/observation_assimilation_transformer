#!/usr/bin/env bash
# Figure 1 (fig:oat): the panels of the OAT scheme.
# The script makes the panels only. The final layout was put together by hand.
# Needs: MeteoSwiss data (panels a, b, d, e). Output: figures/out/fig1_{a,b,c,d,e}.pdf
source "$(dirname "$0")/_lib.sh"

need_swiss
need_cosmo_e1
for panel in a_real b_real c_innovation d_real e_real; do
    run python -m "scripts.plots.oat_scheme.plot_oat_panel_${panel}"
    run cp "data/figures/oat_panel_${panel}.pdf" "$OUT_DIR/fig1_${panel%%_*}.pdf"
done
