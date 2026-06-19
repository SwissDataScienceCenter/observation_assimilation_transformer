#!/bin/bash
#
# Usage:
#   sbatch scripts/submit_evaluate_test_set.sh <checkpoint> [options]
#
# All arguments are forwarded to scripts.evaluate_test_set.
# Example:
#   sbatch scripts/submit_evaluate_test_set.sh \
#       checkpoints/debugging/oat_training_.../best-model.ckpt \
#       --split test --baseline --n-samples 100

#SBATCH --job-name="evaluate_test_set"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

### OUTPUT ###
#SBATCH --output=slurm_logs/evaluate_test_set_%j.log
#SBATCH --error=slurm_logs/evaluate_test_set_%j.err

source .venv/bin/activate

python -m scripts.evaluate_test_set "$@"
