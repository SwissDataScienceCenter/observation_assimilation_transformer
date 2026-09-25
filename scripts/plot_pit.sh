#!/bin/bash
#SBATCH --job-name=plot_pit
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=slurm_logs/plot_pit/plot_pit_%j.log
#SBATCH --error=slurm_logs/plot_pit/plot_pit_%j.err

source .venv/bin/activate

start=$(date +%s)

python -m scripts.plot_pit \
    --model oat_temp_precip \
    --dataset era5_meteostat \
    --split test \
    --n-samples 500 \
    --baseline uninformed_temp_precip

end=$(date +%s)
elapsed=$((end - start))
printf "Duration: %02d:%02d:%02d\n" $((elapsed / 3600)) $((elapsed % 3600 / 60)) $((elapsed % 60))
