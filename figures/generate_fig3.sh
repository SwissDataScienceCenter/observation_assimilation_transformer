#!/usr/bin/env bash
# Figure 3 (fig:example_field): predicted fields in Switzerland.
# Needs: MeteoSwiss data, checkpoint cosmo_peakweather/oat_temp_precip.
# Output: figures/out/fig3.pdf
source "$(dirname "$0")/_lib.sh"

need_swiss
need_ckpt cosmo_peakweather oat_temp_precip
run python -m scripts.plot_assimilation_panel --dataset cosmo_peakweather \
    --time 2023-11-14:16 --n-samples 200 --resolution 250 --out "$OUT_DIR/fig3.pdf"
