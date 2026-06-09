#!/bin/bash
#
# Usage:
#   sbatch scripts/submit_plot_domain.sh <checkpoint> [options]
#
# All arguments are forwarded to scripts.plot_nwp_domain.
# Example:
#   sbatch scripts/submit_plot_domain.sh \
#       checkpoints/debugging/oat_training_.../best-model.ckpt \
#       --var precipitation --n-samples 20 --time 2021-08-04 --split test

#SBATCH --job-name="plot_domain"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

### OUTPUT ###
#SBATCH --output=slurm_logs/plot_domain_%j.log
#SBATCH --error=slurm_logs/plot_domain_%j.err

source .venv/bin/activate

python -m scripts.plot_nwp_domain "$@"
