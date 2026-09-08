#!/bin/bash
#SBATCH --job-name=plot_timeseries
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=slurm_logs/plot_timeseries/plot_timeseries_%j.log
#SBATCH --error=slurm_logs/plot_timeseries/plot_timeseries_%j.err

source .venv/bin/activate

start=$(date +%s)

python -m scripts.plot_timeseries \
    --dataset cosmo_peakweather --model oat_temp_precip \
    --time 2023-09-06:00 --hours 24

end=$(date +%s)
elapsed=$((end - start))
printf "Duration: %02d:%02d:%02d\n" $((elapsed / 3600)) $((elapsed % 3600 / 60)) $((elapsed % 60))
