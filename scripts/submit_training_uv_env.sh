#!/bin/bash
#
# Usage:
#   sbatch scripts/submit_training_uv_env.sh                                    # OAT training (default)
#   sbatch scripts/submit_training_uv_env.sh model=baseline logging=baseline    # Baseline evaluation
#   sbatch scripts/submit_training_uv_env.sh --env uv model=baseline logging=baseline
#   sbatch scripts/submit_training_uv_env.sh dataset.obs_vars=[temperature]     # OAT with config overrides
#
# The --env flag selects the Python environment (poetry or uv).
# All other arguments are forwarded as Hydra overrides to scripts.train_model.

#SBATCH --job-name="oat_training"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=slurm_logs/oat_training/oat_training_%j.log
#SBATCH --error=slurm_logs/oat_training/oat_training_%j.err


ENV="poetry"  # default
HYDRA_ARGS=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --env)
        ENV="$2"
        shift # past argument
        shift # past value
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


srun python -m scripts.train_model "${HYDRA_ARGS[@]}"
