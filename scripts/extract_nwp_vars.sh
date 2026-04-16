#!/bin/bash
#
# Extract NWP variables from the anemoi zarr into a rechunked zarr.
# This is a one-time preprocessing step that reduces per-sample I/O
# from 379 MB to ~3.6 MB (T_2M only) or ~11 MB (T_2M + U_10M + V_10M).
#
# Usage:
#   sbatch scripts/extract_nwp_vars.sh
#   sbatch scripts/extract_nwp_vars.sh --vars T_2M U_10M V_10M

#SBATCH --job-name="extract_nwp_vars"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --no-requeue

### OUTPUT ###
#SBATCH --output=slurm_logs/extract_nwp_vars/extract_nwp_vars_%j.log
#SBATCH --error=slurm_logs/extract_nwp_vars/extract_nwp_vars_%j.err

mkdir -p slurm_logs/extract_nwp_vars

VENV_PATH=$(poetry env info --path)
source "$VENV_PATH/bin/activate"

echo "Starting NWP extraction at $(date)"
echo "Args: $*"

python -m scripts.extract_nwp_vars "$@"

echo "Finished at $(date)"
