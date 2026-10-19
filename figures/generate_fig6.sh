#!/usr/bin/env bash
# Figure 6 (fig:era5_to_cosmo): region hold-out in the western Alps,
# ERA5 / uninformed / OAT compared with COSMO-E1.
# Needs: the Europe dataset and the full COSMO-1E domain (rows 2 and 3).
# Output: figures/out/fig6.pdf
source "$(dirname "$0")/_lib.sh"

need_europe
need_full_domain
need_ckpt era5_meteostat oat_spatial_holdout_temp_precip \
    uninformed_spatial_holdout_temp_precip
run python -m scripts.plot_region_holdout_panel --time 2024-06-21:12 \
    --out "$OUT_DIR/fig6.pdf"
