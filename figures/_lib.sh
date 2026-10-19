# Shared helpers of the figure and table scripts. Source this file, do not run it.
#
# Run a script from the project environment, for example:
#     poetry run bash figures/generate_fig4.sh
# Outputs go to figures/out/. Inputs come from:
#     $OAT_DATA_ROOT (else data), $OAT_CKPT_ROOT (else checkpoints)
# DRY_RUN=1 prints each command and does not run it.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_ROOT="${OAT_DATA_ROOT:-data}"
BENCH_DIR="${OAT_BENCH_DIR:-benchmarking/scaling_benchmark}"
OUT_DIR="figures/out"
mkdir -p "$OUT_DIR"

SWISS_MSG="This part of the figure needs MeteoSwiss data (COSMO-1E and PeakWeather as prepared for the paper). This data is not public, see REPRODUCE.md."
DOWNLOAD_MSG="Get it with: python -m scripts.download_artifacts --what all --dest ."
EUROPE_FILES=(
    era5-europe-mars-n320-2019-2024-1h-v1.zarr
    europe-static.zarr
    hourly_2019_2025_temp_prcp_wspd_wpgt_rhum.parquet
    station_quality.parquet
)
SWISS_ARCHIVE="mch-co1e-an-archive-0p01-2019-2024-1h-v1-pl13.zarr"
FULL_DOMAIN="cosmo-1e_full_domain"
FULL_DOMAIN_MSG="This part needs the full COSMO-1E domain (MeteoSwiss data, not cropped to Switzerland). This data is not public, see REPRODUCE.md."

die() { echo "$(basename "$0"): $*" >&2; exit 1; }

# run COMMAND...: print the command, then run it (not with DRY_RUN=1).
run() {
    echo "+ $*" >&2
    if [[ -z "${DRY_RUN:-}" ]]; then "$@"; fi
}

# need PATH MESSAGE: stop when PATH does not exist.
need() { [[ -e "$1" ]] || die "missing input $1. $2"; }

need_europe() {
    local f
    for f in "${EUROPE_FILES[@]}"; do need "$DATA_ROOT/$f" "$DOWNLOAD_MSG"; done
}

has_swiss() {
    [[ -e "$DATA_ROOT/cosmo-1e_rechunked" && -e "$DATA_ROOT/peakweather" \
        && -e "$DATA_ROOT/$SWISS_ARCHIVE" ]]
}
need_swiss() {
    has_swiss || die "missing input $DATA_ROOT/cosmo-1e_rechunked, $DATA_ROOT/peakweather or $DATA_ROOT/$SWISS_ARCHIVE. $SWISS_MSG"
}
has_full_domain() {
    [[ -e "$DATA_ROOT/$FULL_DOMAIN/$SWISS_ARCHIVE" \
        && -e "$DATA_ROOT/$FULL_DOMAIN/nwp_extracted.zarr" ]]
}
need_full_domain() {
    has_full_domain || die "missing input $DATA_ROOT/$FULL_DOMAIN/$SWISS_ARCHIVE or $DATA_ROOT/$FULL_DOMAIN/nwp_extracted.zarr. $FULL_DOMAIN_MSG"
}

# ckpt DATASET MODEL: print the checkpoint path of a registry entry.
ckpt() {
    python -c 'import sys, yaml
from oat.paths import REGISTRY_PATH, checkpoint_root
print(checkpoint_root() / yaml.safe_load(REGISTRY_PATH.read_text())[sys.argv[1]][sys.argv[2]])' "$1" "$2"
}

# need_ckpt DATASET MODEL...: stop when a registry checkpoint is not on disk.
need_ckpt() {
    local dataset="$1" model path
    shift
    for model in "$@"; do
        path="$(ckpt "$dataset" "$model")"
        need "$path" "$DOWNLOAD_MSG"
    done
}

need_gpu() {
    [[ -n "${DRY_RUN:-}" ]] && return 0
    python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' \
        || die "missing input: a CUDA GPU. On a cluster, run: sbatch --wrap \"poetry run bash $0\""
}

# newest LABEL MODE [SUFFIX]: print the newest benchmark run folder, or stop.
# Assign the result to a variable: set -e does not see a failure inside the
# arguments of another command.
newest() {
    local runs=("$BENCH_DIR"/????-??-??_??-??-??_"$1"_"$2"_B1"${3:-}")
    [[ -d "${runs[-1]}" ]] \
        || die "missing input: benchmark run $1 ($2) in $BENCH_DIR. Run figures/generate_figA1.sh first."
    echo "${runs[-1]}"
}
