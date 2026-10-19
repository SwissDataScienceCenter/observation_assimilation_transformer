#!/usr/bin/env bash
# Figure 9 (fig:fullattention_vs_sparse, 4 panels): OAT with sparse and with
# full attention. It uses the OAT and GP runs of figure 8 and stops when they
# are missing. Needs: a CUDA GPU.
# Output: figures/out/fig9_{fixed,proportional}_{train,inference}.pdf
source "$(dirname "$0")/_lib.sh"

need_ckpt cosmo_peakweather oat_temp_precip gp_oat_temp
OAT_CKPT="$(ckpt cosmo_peakweather oat_temp_precip)"
GP_WEIGHTS="$(ckpt cosmo_peakweather gp_oat_temp)"
OAT_CONFIG="$(dirname "$OAT_CKPT")/config.yaml"
GP_CONFIG="$(dirname "$GP_WEIGHTS")/config.yaml"
need "$OAT_CONFIG" "$DOWNLOAD_MSG"
need "$GP_CONFIG" "$DOWNLOAD_MSG"
for mode in fixed proportional; do  # stop early when the figure 8 runs are missing
    oat_run="$(newest oat "$mode")"
    gp_exact="$(newest gp_oat "$mode" _exact)"
    gp_iterative="$(newest gp_oat "$mode")"
done
need_gpu

for mode in fixed proportional; do
    run python -m scripts.benchmark_station_scaling --model oat \
        --label oat_full_attention --query-mode "$mode" --config "$OAT_CONFIG" \
        --dataset-config "$GP_CONFIG" --override model.masking.mode=full \
        --output-root "$BENCH_DIR"
done

for mode in fixed proportional; do
    oat_run="$(newest oat "$mode")"
    gp_exact="$(newest gp_oat "$mode" _exact)"
    gp_iterative="$(newest gp_oat "$mode")"
    full_run="$(newest oat_full_attention "$mode")"
    for step in train inference; do
        run python -m scripts.plot_station_scaling --oat "$oat_run" \
            --gp-exact "$gp_exact" --gp-iterative "$gp_iterative" \
            --oat-full-attention "$full_run" --step "$step" --figure-dir "$OUT_DIR"
        run mv "$OUT_DIR/fullattention_vs_sparse_${mode}_${step}.pdf" \
            "$OUT_DIR/fig9_${mode}_${step}.pdf"
        run rm -f "$OUT_DIR/fullattention_vs_sparse_${mode}_${step}.png" \
            "$OUT_DIR/station_scaling_${mode}_${step}.pdf" \
            "$OUT_DIR/station_scaling_${mode}_${step}.png"
    done
done
