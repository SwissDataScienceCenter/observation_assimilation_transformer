#!/bin/bash
#SBATCH --job-name=plot_extreme_events
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=slurm_logs/plot_extreme_events/plot_extreme_events_%j.log
#SBATCH --error=slurm_logs/plot_extreme_events/plot_extreme_events_%j.err

# Usage: sbatch scripts/plot_extreme_events.sh [--list-candidates]
# SLURM does not make the log directory. Make it before sbatch:
#   mkdir -p slurm_logs/plot_extreme_events

source .venv/bin/activate

start=$(date +%s)

python -m scripts.plot_extreme_events --n-samples 200 --map-n-samples 50 "$@"

end=$(date +%s)
elapsed=$((end - start))
printf "Duration: %02d:%02d:%02d\n" $((elapsed / 3600)) $((elapsed % 3600 / 60)) $((elapsed % 60))
