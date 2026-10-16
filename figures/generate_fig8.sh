#!/usr/bin/env bash
# Figure 8 (fig:scaling, 4 panels): train and inference time of OAT and the GP
# for more and more stations. The benchmark uses fake station data, so it
# needs no dataset. It reads the model configs and the GP weights of the Swiss
# checkpoints. Needs: a CUDA GPU (about 4 h together with figure 9).
# Timings depend on the hardware.
# Output: figures/out/fig8_{fixed,proportional}_{train,inference}.pdf
source "$(dirname "$0")/_lib.sh"

need_ckpt cosmo_peakweather oat_temp_precip gp_oat_temp
OAT_CKPT="$(ckpt cosmo_peakweather oat_temp_precip)"
GP_WEIGHTS="$(ckpt cosmo_peakweather gp_oat_temp)"
OAT_CONFIG="$(dirname "$OAT_CKPT")/config.yaml"
GP_CONFIG="$(dirname "$GP_WEIGHTS")/config.yaml"
need "$OAT_CONFIG" "$DOWNLOAD_MSG"
need "$GP_CONFIG" "$DOWNLOAD_MSG"
need_gpu

for mode in fixed proportional; do
    run python -m scripts.benchmark_station_scaling --model oat --label oat \
        --query-mode "$mode" --config "$OAT_CONFIG" --dataset-config "$GP_CONFIG" \
        --output-root "$BENCH_DIR"
    run python -m scripts.benchmark_station_scaling --model gp_baseline \
        --label gp_oat --query-mode "$mode" --config "$GP_CONFIG" \
        --checkpoint "$GP_WEIGHTS" --output-root "$BENCH_DIR"
    run python -m scripts.benchmark_station_scaling --model gp_baseline \
        --label gp_oat --query-mode "$mode" --config "$GP_CONFIG" \
        --checkpoint "$GP_WEIGHTS" --gp-exact-cholesky --output-root "$BENCH_DIR"
done

for mode in fixed proportional; do
    oat_run="$(newest oat "$mode")"
    gp_exact="$(newest gp_oat "$mode" _exact)"
    gp_iterative="$(newest gp_oat "$mode")"
    for step in train inference; do
        run python -m scripts.plot_station_scaling --oat "$oat_run" \
            --gp-exact "$gp_exact" --gp-iterative "$gp_iterative" \
            --step "$step" --figure-dir "$OUT_DIR"
        run mv "$OUT_DIR/station_scaling_${mode}_${step}.pdf" \
            "$OUT_DIR/fig8_${mode}_${step}.pdf"
        run rm -f "$OUT_DIR/station_scaling_${mode}_${step}.png"
    done
done
