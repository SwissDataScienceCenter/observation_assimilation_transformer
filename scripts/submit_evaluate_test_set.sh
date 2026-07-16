#!/bin/bash
#
# Submits one SLURM evaluation job per entry in configs/checkpoints.yaml.
#
# Usage:
#   bash scripts/submit_evaluate_test_set.sh
#   bash scripts/submit_evaluate_test_set.sh --dataset cosmo_peakweather
#   bash scripts/submit_evaluate_test_set.sh --dataset cosmo_peakweather --model oat_temp_precip
#   bash scripts/submit_evaluate_test_set.sh --env uv
#
# Any unknown flags (e.g. --split val, --n-samples 50) are forwarded to
# evaluate_test_set.py for every submitted job.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="$PROJECT_ROOT/configs/checkpoints.yaml"

FILTER_DATASET=""
FILTER_MODEL=""
ENV="poetry"
EVAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) FILTER_DATASET="$2"; shift 2 ;;
        --model)   FILTER_MODEL="$2";   shift 2 ;;
        --env)     ENV="$2";            shift 2 ;;
        *)         EVAL_ARGS+=("$1");   shift ;;
    esac
done

if [ "$ENV" = "poetry" ]; then
    VENV_PATH=$(cd "$PROJECT_ROOT" && poetry env info --path)
elif [ "$ENV" = "uv" ]; then
    VENV_PATH="$PROJECT_ROOT/.venv"
else
    echo "Unknown environment: $ENV" >&2; exit 1
fi

PAIRS=$(python3 - <<PYEOF
import yaml
with open("$REGISTRY") as f:
    registry = yaml.safe_load(f)
for dataset, models in (registry or {}).items():
    if "$FILTER_DATASET" and dataset != "$FILTER_DATASET":
        continue
    models = models or {}
    has_ckpt = False
    for model, path in models.items():
        if path is not None:
            has_ckpt = True
        if "$FILTER_MODEL" and model != "$FILTER_MODEL":
            continue
        if path is not None:
            print(f"{dataset}\t{model}")
    # Weightless interpolator baseline: one job per dataset (needs a sibling
    # checkpoint to borrow the data config from).
    if has_ckpt and (not "$FILTER_MODEL" or "$FILTER_MODEL" == "interpolator"):
        print(f"{dataset}\tinterpolator")
PYEOF
)

if [[ -z "$PAIRS" ]]; then
    echo "No checkpoints to submit (check filters and null entries in $REGISTRY)" >&2
    exit 1
fi

mkdir -p "$PROJECT_ROOT/slurm_logs/evaluation"

EXTRA="${EVAL_ARGS[*]}"

while IFS=$'\t' read -r dataset model; do
    log_prefix="$PROJECT_ROOT/slurm_logs/evaluation/${dataset}_${model}"
    job_id=$(sbatch --parsable \
        --job-name="eval_${dataset}_${model}" \
        --partition=debug \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=0:30:00 \
        --output="${log_prefix}_%j.log" \
        --error="${log_prefix}_%j.err" \
        --wrap="source '${VENV_PATH}/bin/activate' && cd '${PROJECT_ROOT}' && python -m scripts.evaluate_test_set --dataset '${dataset}' --model '${model}' ${EXTRA}")
    echo "Submitted job ${job_id}: ${dataset} / ${model}"
done <<< "$PAIRS"
