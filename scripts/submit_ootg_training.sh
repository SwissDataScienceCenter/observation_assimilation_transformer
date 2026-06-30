#!/bin/bash
#
# Usage:
#   sbatch scripts/submit_ootg_training.sh                                      # COSMO temp (default)
#   sbatch scripts/submit_ootg_training.sh dataset=cosmo_peakweather/temp_precip
#   sbatch scripts/submit_ootg_training.sh dataset=era5_meteostat/temp
#   sbatch scripts/submit_ootg_training.sh dataset=era5_meteostat/temp_precip
#   sbatch scripts/submit_ootg_training.sh --env uv dataset=era5_meteostat/temp num_layers=8
#
# The --env flag selects the Python environment (poetry or uv).
# All other arguments are forwarded as Hydra overrides to scripts.train_ootg_baseline.

#SBATCH --job-name="ootg_baseline"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --no-requeue

### OUTPUT ###
#SBATCH --output=slurm_logs/ootg_training/ootg_training_%j.log
#SBATCH --error=slurm_logs/ootg_training/ootg_training_%j.err


ENV="poetry"  # default
HYDRA_ARGS=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --env)
        ENV="$2"
        shift
        shift
        ;;
        *)
        HYDRA_ARGS+=("$1")
        shift
        ;;
    esac
done

echo "Using environment: $ENV"
echo "Hydra overrides: ${HYDRA_ARGS[*]}"

if [ "$ENV" = "uv" ]; then
    echo "Activating UV venv"
    source .venv/bin/activate
elif [ "$ENV" = "poetry" ]; then
    echo "Activating Poetry environment"
    VENV_PATH=$(poetry env info --path)
    source "$VENV_PATH/bin/activate"
else
    echo "Unknown environment: $ENV"
    exit 1
fi

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
python -c "import torch; print(torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

echo "Checking CUDA availability..."
python - <<'EOF'
import torch
if not torch.cuda.is_available():
    print("❌ CUDA is NOT available. Check your torch installation! Exiting job.")
    exit(1)
print("✅ CUDA is available.")
EOF

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python -m scripts.train_ootg_baseline "${HYDRA_ARGS[@]}"
