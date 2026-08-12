#!/bin/bash
#
# Usage:
#   sbatch scripts/submit_profile_memory.sh [hydra overrides...]
#
# Example:
#   sbatch scripts/submit_profile_memory.sh \
#     dataset=era5_meteostat/full \
#     dataloader.train.batch_size=32 \
#     dataloader.train.num_workers=2 \
#     model.d_model=32

#SBATCH --job-name="oat_profile"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=slurm_logs/oat_training/oat_profile_%j.log
#SBATCH --error=slurm_logs/oat_training/oat_profile_%j.err

VENV_PATH=$(poetry env info --path)
source "$VENV_PATH/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
python -c "import torch; print(torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

srun python -m scripts.profile_memory "$@"
