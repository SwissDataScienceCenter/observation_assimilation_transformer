#!/usr/bin/env bash
# Figure 1 (fig:oat): the panels of the OAT scheme.
# The script makes the panels only. The final layout was put together by hand.
# Needs: the Swiss data (the crop to Switzerland is enough).
# Panel a uses the zoom image of the panel script, not the full-domain image.
# Output: figures/out/fig1_{a,b,c,d,e}.pdf
source "$(dirname "$0")/_lib.sh"

need_swiss
for panel in a_real b_real c_innovation d_real e_real; do
    run python -m "scripts.plots.oat_scheme.plot_oat_panel_${panel}"
    src="oat_panel_${panel}"
    [[ "$panel" == a_real ]] && src="oat_panel_a_zoom"
    run cp "data/figures/${src}.pdf" "$OUT_DIR/fig1_${panel%%_*}.pdf"
done
