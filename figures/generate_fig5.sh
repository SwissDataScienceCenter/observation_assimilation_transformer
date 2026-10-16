#!/usr/bin/env bash
# Figure 5 (fig:error_cosmo, fig:error_era5): CRPS by season, elevation and
# distance to the nearest station. It needs the per-station, per-time dumps
# of evaluate_test_set --long. The script computes a missing dump (CPU, about
# 30 min for each model). The Europe part comes first. The Swiss part needs
# MeteoSwiss data. Output: figures/out/fig5_europe.pdf, fig5_switzerland.pdf
source "$(dirname "$0")/_lib.sh"

MODELS=(oat_temp_precip uninformed_temp_precip gp_oat_temp gp_oat_precip)

stratification() {
    local dataset="$1" name="$2" missing model
    need_ckpt "$dataset" "${MODELS[@]}"
    missing="$(python -c 'import sys
from scripts.plot_error_stratification import long_parquet_path
print(" ".join(m for m in sys.argv[2:]
               if not long_parquet_path(sys.argv[1], m, "test").exists()))' \
        "$dataset" "${MODELS[@]}")"
    for model in $missing; do
        run python -m scripts.evaluate_test_set --dataset "$dataset" \
            --model "$model" --long
    done
    run python -m scripts.plot_error_stratification --dataset "$dataset" \
        --model oat_temp_precip \
        --baselines uninformed_temp_precip,gp_oat_temp,gp_oat_precip \
        --metric crps --split test --out "$OUT_DIR/fig5_$name"
}

need_europe
stratification era5_meteostat europe
need_swiss
stratification cosmo_peakweather switzerland
