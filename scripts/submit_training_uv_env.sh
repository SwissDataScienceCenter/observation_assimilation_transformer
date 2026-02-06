#!/bin/bash

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


source .venv/bin/activate


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


srun python -m scripts.train_model