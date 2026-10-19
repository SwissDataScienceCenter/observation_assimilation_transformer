#!/usr/bin/env bash
# Table 1 (tab:ml_results) and its confidence intervals (tab:ci_table).
# Europe rows: evaluated again from the released checkpoints and data (CPU,
# about 4 h in total). The evaluations write data/results/<tag>_metrics_test.*
# and replace the released files of the same name.
# Swiss rows, the Swiss OOTG row and the COSMO interpolation row of the region
# hold-out: these rows then come from the released result files (python -m
# scripts.download_artifacts --what checkpoints), unless the Swiss data is on
# disk. The Swiss OOTG row and the COSMO row need the full COSMO-1E domain.
# Output: figures/out/table1_{means,ci}.{tex,txt}
source "$(dirname "$0")/_lib.sh"

EUROPE_MODELS=(oat_temp_precip uninformed_temp_precip ootg_temp_precip
    gp_oat_temp gp_oat_precip deep_gp_oat_temp deep_gp_oat_precip)
HOLDOUT_MODELS=(oat_spatial_holdout_temp_precip uninformed_spatial_holdout_temp_precip)
SWISS_MODELS=(oat_temp_precip oat_vanilla_temp_precip oat_tetnp_temp_precip
    uninformed_temp_precip gp_oat_temp gp_oat_precip
    deep_gp_oat_temp deep_gp_oat_precip)
HOLDOUT=(--split-config spatial_region_holdout)

evaluate() { run python -m scripts.evaluate_test_set --dataset "$@"; }

need_europe
need_ckpt era5_meteostat "${EUROPE_MODELS[@]}" "${HOLDOUT_MODELS[@]}"
for model in interpolator "${EUROPE_MODELS[@]}"; do
    evaluate era5_meteostat --model "$model"
done
for model in interpolator "${HOLDOUT_MODELS[@]}"; do
    evaluate era5_meteostat --model "$model" "${HOLDOUT[@]}"
done
if has_full_domain; then
    evaluate era5_meteostat --model cosmo_interpolator "${HOLDOUT[@]}"
else
    echo "$(basename "$0"): note: no full COSMO-1E domain on disk. The Swiss OOTG row and the COSMO holdout row come from the released result files." >&2
fi
if has_swiss; then
    need_ckpt cosmo_peakweather "${SWISS_MODELS[@]}"
    for model in interpolator "${SWISS_MODELS[@]}"; do
        evaluate cosmo_peakweather --model "$model"
    done
    if has_full_domain; then
        need_ckpt cosmo_peakweather ootg_temp_precip
        evaluate cosmo_peakweather --model ootg_temp_precip
    fi
fi
run python -m scripts.build_results_table --out-prefix "$OUT_DIR/table1"
