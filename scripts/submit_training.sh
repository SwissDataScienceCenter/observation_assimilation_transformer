#!/bin/bash
#
# Unified SLURM submission script for all model/dataset training combos.
# Not meant to be sbatch'd directly by hand - use the dispatcher instead:
#
#   python -m scripts.train --model oat --dataset cosmo_peakweather/temp_precip --submit
#
# The dispatcher computes a systematic --job-name/--output/--error and passes
# them on the `sbatch` command line (overriding the placeholders below), then
# forwards --model/--dataset/extra Hydra overrides to this script, which just
# activates the environment and hands off to `scripts.train` inside the job.
#
# The --env flag selects the Python environment (poetry or uv).

#SBATCH --job-name="training"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=slurm_logs/training/training_%j.log
#SBATCH --error=slurm_logs/training/training_%j.err


ENV="poetry"  # default
ARGS=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --env)
        ENV="$2"
        shift # past argument
        shift # past value
        ;;
        *)
        ARGS+=("$1")
        shift
        ;;
    esac
done

echo "Using environment: $ENV"
echo "Dispatcher args: ${ARGS[*]}"

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

srun python -m scripts.train "${ARGS[@]}"
