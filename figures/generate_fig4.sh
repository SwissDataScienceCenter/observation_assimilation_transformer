#!/usr/bin/env bash
# Figure 4 (fig:pit): PIT histograms (calibration), Europe and Switzerland.
# The Europe part comes first. The Swiss part needs MeteoSwiss data.
# Output: figures/out/fig4_europe.pdf, figures/out/fig4_switzerland.pdf
source "$(dirname "$0")/_lib.sh"

pit() {
    run python -m scripts.plot_pit --model oat_temp_precip --dataset "$1" \
        --split test --n-samples 500 --baseline uninformed_temp_precip \
        --out "$OUT_DIR/fig4_$2"
}

need_europe
need_ckpt era5_meteostat oat_temp_precip uninformed_temp_precip
pit era5_meteostat europe
need_swiss
need_ckpt cosmo_peakweather oat_temp_precip uninformed_temp_precip
pit cosmo_peakweather switzerland
