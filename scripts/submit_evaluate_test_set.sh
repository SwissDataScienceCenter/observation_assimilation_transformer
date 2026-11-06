#!/bin/bash
#
# Submits one SLURM evaluation job per entry in configs/checkpoints.yaml.
#
# Usage:
#   bash scripts/submit_evaluate_test_set.sh
#   bash scripts/submit_evaluate_test_set.sh --dataset cosmo_peakweather
#   bash scripts/submit_evaluate_test_set.sh --dataset cosmo_peakweather --model oat_temp_precip
#   bash scripts/submit_evaluate_test_set.sh --env uv
#   bash scripts/submit_evaluate_test_set.sh --missing
#   bash scripts/submit_evaluate_test_set.sh --missing --dataset era5_meteostat
#
# --missing submits jobs only for entries without a result CSV in data/results.
# It also sets --split-config spatial_region_holdout for the holdout models.
#
# Any unknown flags (e.g. --split val, --n-samples 50) are forwarded to
# evaluate_test_set.py for every submitted job.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="$PROJECT_ROOT/configs/checkpoints.yaml"

FILTER_DATASET=""
FILTER_MODEL=""
ENV="poetry"
MISSING=""
EVAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) FILTER_DATASET="$2"; shift 2 ;;
        --model)   FILTER_MODEL="$2";   shift 2 ;;
        --env)     ENV="$2";            shift 2 ;;
        --missing) MISSING=1;           shift ;;
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

if [[ -n "$MISSING" ]]; then
    if [[ -n "$FILTER_MODEL" ]]; then
        echo "Do not use --missing with --model" >&2; exit 1
    fi
    # The other flags (for example --split val) change which result file counts.
    PAIRS=$(cd "$PROJECT_ROOT" && "$VENV_PATH/bin/python" -m scripts.evaluate_test_set \
        --list-missing ${FILTER_DATASET:+--dataset "$FILTER_DATASET"} "${EVAL_ARGS[@]}") \
        || { echo "Could not list the missing runs" >&2; exit 1; }
    if [[ -z "$PAIRS" ]]; then
        echo "All registry entries have results. Nothing to submit." >&2
        exit 0
    fi
else
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
    # checkpoint to borrow the data config from). Skip when the registry
    # already declares its own "interpolator" entry -- the loop above
    # already emitted it, so adding it again would submit a duplicate job.
    if (
        "interpolator" not in models
        and has_ckpt
        and (not "$FILTER_MODEL" or "$FILTER_MODEL" == "interpolator")
    ):
        print(f"{dataset}\tinterpolator")
PYEOF
)
fi

if [[ -z "$PAIRS" ]]; then
    echo "No checkpoints to submit (check filters and null entries in $REGISTRY)" >&2
    exit 1
fi

mkdir -p "$PROJECT_ROOT/slurm_logs/evaluation"

EXTRA="${EVAL_ARGS[*]}"

while IFS=$'\t' read -r dataset model split_config; do
    split_arg="${split_config:+--split-config '${split_config}'}"
    log_prefix="$PROJECT_ROOT/slurm_logs/evaluation/${dataset}_${model}"
    job_id=$(sbatch --parsable \
        --job-name="eval_${dataset}_${model}" \
        --partition=normal \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=02:00:00 \
        --output="${log_prefix}_%j.log" \
        --error="${log_prefix}_%j.err" \
        --wrap="source '${VENV_PATH}/bin/activate' && cd '${PROJECT_ROOT}' && python -m scripts.evaluate_test_set --dataset '${dataset}' --model '${model}' ${split_arg} ${EXTRA}")
    echo "Submitted job ${job_id}: ${dataset} / ${model}"
done <<< "$PAIRS"
