"""Central dispatcher for OAT/GP/OOTG training runs.

Picks the right entry-point script and Hydra overrides for a given
model/dataset combination, and gives every run a systematic, recognizable
run name used consistently for the checkpoint dir and W&B run.

Usage:
    python -m scripts.train --model oat --dataset cosmo_peakweather/temp_precip
    python -m scripts.train --model gp --dataset era5_meteostat/temp --dry-run
    python -m scripts.train --model ootg --dataset cosmo_peakweather/temp --submit
    python -m scripts.train --model oat --dataset cosmo_peakweather/temp \
        trainer.max_epochs=1
"""

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

DATASET_FAMILIES = ("era5_meteostat", "cosmo_peakweather")
DATASET_VARIANTS = ("temp", "precip", "temp_precip")
DATASETS = tuple(
    f"{family}/{variant}" for family in DATASET_FAMILIES for variant in DATASET_VARIANTS
)
UNIVARIATE_VARIANTS = ("temp", "precip")


@dataclass(frozen=True)
class ModelSpec:
    entry_module: str
    overrides: tuple[str, ...] = ()
    univariate_only: bool = False


MODEL_SPECS = {
    "oat": ModelSpec("scripts.train_model", ("model=oat", "predictor=oat_predictor")),
    "uninformed_oat": ModelSpec(
        "scripts.train_model",
        ("model=uninformed_oat", "predictor=oat_predictor", "split=no_conditioning"),
    ),
    "gp": ModelSpec(
        "scripts.train_model",
        ("model=gp_baseline", "predictor=gp_predictor"),
        univariate_only=True,
    ),
    "deep_gp": ModelSpec(
        "scripts.train_model",
        ("model=deep_gp_baseline", "predictor=gp_predictor"),
        univariate_only=True,
    ),
    "ootg": ModelSpec("scripts.train_ootg_baseline"),
}


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit via sbatch instead of running now.",
    )
    parser.add_argument(
        "--env",
        default="poetry",
        choices=("poetry", "uv"),
        help="Python env for --submit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without running it.",
    )
    namespace, extra = parser.parse_known_args(argv)
    return namespace, extra


def validate(model: str, dataset: str) -> None:
    _, variant = dataset.split("/", 1)
    if MODEL_SPECS[model].univariate_only and variant not in UNIVARIATE_VARIANTS:
        raise SystemExit(
            f"Model '{model}' is univariate-only (see gp_baseline.py); dataset variant "
            f"'{variant}' is multivariate. Use one of {UNIVARIATE_VARIANTS} instead."
        )


def run_tag(model: str, dataset: str) -> str:
    family, variant = dataset.split("/", 1)
    return f"{model}_{family}-{variant}"


def compute_run_name(model: str, dataset: str, now_str: str, job_id: str) -> str:
    return f"{run_tag(model, dataset)}_{now_str}_job{job_id}"


def override_key(override: str) -> str:
    return override.split("=", 1)[0].lstrip("+")


def build_overrides(
    model: str, dataset: str, run_name: str, extra: Sequence[str]
) -> list[str]:
    spec = MODEL_SPECS[model]
    computed = [
        *spec.overrides,
        f"dataset={dataset}",
        f"logging.group={model}",
        f"run_name={run_name}",
    ]
    user_keys = {override_key(o) for o in extra}
    kept = [o for o in computed if override_key(o) not in user_keys]
    return [*kept, *extra]


def build_command(
    model: str,
    dataset: str,
    run_name: str,
    extra: Sequence[str],
    submit: bool,
    env: str,
) -> list[str]:
    if submit:
        tag = run_tag(model, dataset)
        return [
            "sbatch",
            f"--job-name={tag}",
            f"--output=slurm_logs/training/{tag}_%j.log",
            f"--error=slurm_logs/training/{tag}_%j.err",
            "scripts/submit_training.sh",
            "--env",
            env,
            "--model",
            model,
            "--dataset",
            dataset,
            *extra,
        ]
    overrides = build_overrides(model, dataset, run_name, extra)
    return [sys.executable, "-m", MODEL_SPECS[model].entry_module, *overrides]


def main(argv: Sequence[str] | None = None) -> None:
    args, extra = parse_args(argv if argv is not None else sys.argv[1:])
    validate(args.model, args.dataset)

    now_str = datetime.now().strftime("%Y%m%d")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_name = compute_run_name(args.model, args.dataset, now_str, job_id)

    command = build_command(
        args.model, args.dataset, run_name, extra, args.submit, args.env
    )
    print(f"Resolved command: {' '.join(command)}")
    if args.dry_run:
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
