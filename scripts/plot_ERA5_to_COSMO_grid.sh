#!/bin/bash
#SBATCH --job-name=plot_ERA5_to_COSMO_grid
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=slurm_logs/plot_ERA5_to_COSMO_grid/plot_ERA5_to_COSMO_grid_%j.log
#SBATCH --error=slurm_logs/plot_ERA5_to_COSMO_grid/plot_ERA5_to_COSMO_grid_%j.err

source .venv/bin/activate

start=$(date +%s)

python -m scripts.plot_ERA5_to_COSMO_grid \
    '/scratch/mch/cdonner/observation_assimilation_transformer/checkpoints/debugging/oat_training_20260615_job255139/best-model-epoch=15-val_loss=0.5021.ckpt' \
    --var prcp \
    --time 2024-06-21:12 \
    --nwp-var TOT_PREC_1H \
    --era5-var tp

# For wind:
    # --var wspd \
    # --time 2024-06-21:12 \
    # --nwp-var wind_speed \
    # --era5-var wind_speed


end=$(date +%s)
elapsed=$((end - start))
printf "Duration: %02d:%02d:%02d\n" $((elapsed / 3600)) $((elapsed % 3600 / 60)) $((elapsed % 60))
