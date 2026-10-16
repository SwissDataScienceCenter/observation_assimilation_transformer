#!/usr/bin/env bash
# Figure 7 (fig:extreme_events): Maggia flood (Switzerland) and heat wave (Europe).
# Without MeteoSwiss data the script makes the Europe row only, then stops.
# Output: figures/out/fig7.pdf (both rows) or figures/out/fig7_europe.pdf
source "$(dirname "$0")/_lib.sh"

need_europe
need_ckpt era5_meteostat oat_temp_precip
if ! has_swiss; then
    run python -m scripts.plot_extreme_events --n-samples 200 --map-n-samples 50 \
        --dataset era5_meteostat --out "$OUT_DIR/fig7_europe.pdf"
    need_swiss
fi
need_ckpt cosmo_peakweather oat_temp_precip
run python -m scripts.plot_extreme_events --n-samples 200 --map-n-samples 50 \
    --out "$OUT_DIR/fig7.pdf"
