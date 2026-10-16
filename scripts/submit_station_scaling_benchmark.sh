#!/bin/bash
#
# Station-scaling benchmark of OAT and GPBaseline (scripts/benchmark_station_scaling.py).
# Every run writes its own folder to benchmarking/scaling_benchmark/.
#
# Usage (from the repository root):
#   sbatch scripts/submit_station_scaling_benchmark.sh

#SBATCH --job-name="oat_scaling_benchmark"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=benchmarking/logs/scaling_benchmark_%j.log
#SBATCH --error=benchmarking/logs/scaling_benchmark_%j.err


echo "Activating UV venv"
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


# The runs go one after the other, each on the first GPU only (--device cuda
# means cuda:0), so they do not compete for the node and the timings stay
# comparable. The other GPUs of the node stay idle.

# Models as in the paper: the registry entries of configs/checkpoints.yaml
# (COSMO/PeakWeather), below $OAT_CKPT_ROOT (else <repo>/checkpoints).
# The data stays fake.
ckpt() {
    python -c "from oat.predictor_module.checkpoint_utils import resolve_checkpoint; print(resolve_checkpoint('cosmo_peakweather', '$1'))"
}
OAT_CONFIG="$(dirname "$(ckpt oat_temp_precip)")/config.yaml"
GP_OAT_WEIGHTS="$(ckpt gp_oat_temp)"
GP_OAT_CONFIG="$(dirname "$GP_OAT_WEIGHTS")/config.yaml"
DEEP_GP_OAT_WEIGHTS="$(ckpt deep_gp_oat_temp)"
DEEP_GP_OAT_CONFIG="$(dirname "$DEEP_GP_OAT_WEIGHTS")/config.yaml"

# Same inputs and outputs for all models: the dataset part of the GP config
# (NWP T_2M and T_G, 5 topography channels -> temperature). The GPs already
# have it; OAT takes it with --dataset-config.
TASK_CONFIG="$GP_OAT_CONFIG"

for QUERY_MODE in fixed proportional; do
    # fixed:        1024 query stations, isolates the cost of adding stations.
    # proportional: N_query = N_obs / 4, the 80/20 split used in training.
    srun python -m scripts.benchmark_station_scaling --model oat --label oat --query-mode "$QUERY_MODE" \
        --config "$OAT_CONFIG" --dataset-config "$TASK_CONFIG"

    # # GPs: approximate (gpytorch default, CG above 800 points) and exact Cholesky,
    # # with the trained weights (the CG cost depends on the trained conditioning).
    srun python -m scripts.benchmark_station_scaling --model gp_baseline --label gp_oat --query-mode "$QUERY_MODE" \
        --config "$GP_OAT_CONFIG" --checkpoint "$GP_OAT_WEIGHTS"
    srun python -m scripts.benchmark_station_scaling --model gp_baseline --label gp_oat --query-mode "$QUERY_MODE" \
        --config "$GP_OAT_CONFIG" --checkpoint "$GP_OAT_WEIGHTS" --gp-exact-cholesky
    srun python -m scripts.benchmark_station_scaling --model gp_baseline --label deep_gp_oat --query-mode "$QUERY_MODE" \
        --config "$DEEP_GP_OAT_CONFIG" --checkpoint "$DEEP_GP_OAT_WEIGHTS"
    srun python -m scripts.benchmark_station_scaling --model gp_baseline --label deep_gp_oat --query-mode "$QUERY_MODE" \
        --config "$DEEP_GP_OAT_CONFIG" --checkpoint "$DEEP_GP_OAT_WEIGHTS" --gp-exact-cholesky

    # OAT with full attention: no neighbour search, dense unmasked attention.
    srun python -m scripts.benchmark_station_scaling --model oat --label oat_full_attention --query-mode "$QUERY_MODE" \
        --config "$OAT_CONFIG" --dataset-config "$TASK_CONFIG" --override model.masking.mode=full
done
