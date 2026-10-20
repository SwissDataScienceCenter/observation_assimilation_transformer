#!/usr/bin/env bash
# Figure 2 (fig:data): station splits of both datasets and the region hold-out.
# Needs: MeteoSwiss data and the Europe dataset. Output: figures/out/fig2.pdf
source "$(dirname "$0")/_lib.sh"

need_europe
need_swiss
run python -m scripts.plot_dataset_stations --region-holdout --seed 42 \
    --out "$OUT_DIR/fig2.pdf"
