#!/bin/bash
#SBATCH --job-name=plot_pit
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=slurm_logs/plot_pit/plot_pit_%j.log
#SBATCH --error=slurm_logs/plot_pit/plot_pit_%j.err

source .venv/bin/activate

start=$(date +%s)

python -m scripts.plot_pit \
    '/scratch/mch/cdonner/observation_assimilation_transformer/checkpoints/debugging/oat_training_20260615_job255139/best-model-epoch=15-val_loss=0.5021.ckpt' \
    --split test \
    --n-samples 200 \
    --n-timesteps 100 

end=$(date +%s)
elapsed=$((end - start))
printf "Duration: %02d:%02d:%02d\n" $((elapsed / 3600)) $((elapsed % 3600 / 60)) $((elapsed % 60))

# '/scratch/mch/cdonner/observation_assimilation_transformer/checkpoints/oat_L/oat_L_v1_20260619_job4663176/best-model-epoch=72-val_loss=0.2125.ckpt' \

