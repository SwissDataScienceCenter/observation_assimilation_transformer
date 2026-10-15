"""
Station-scaling benchmark: compute cost of OAT and GPBaseline vs number of stations.

The models get fake (random) inputs in exactly the format the predictor hands
them after ``field_batch_utils._unwrap_fields_to_model_batch``. No data is
loaded; only the Hydra configs are read to build the models.

In training, the split into observed and query stations is done by
``TrainDataset`` / ``EvalDataset`` (data_split.py) and the predictor pops the
query targets; ``make_fake_batch`` below plays both roles.

For every number of observed stations N_obs we time
  - "train":     forward_with_loss + backward
  - "inference": forward under torch.no_grad()
  - "knn":       OAT only, the neighbour searches of OAT._forward on their own
                 (already included in "train" and "inference")
and record the median wall time and the peak GPU memory.

Usage:
    python -m scripts.benchmark_station_scaling --model oat
    python -m scripts.benchmark_station_scaling --model gp_baseline --query-mode proportional
    python -m scripts.benchmark_station_scaling --model gp_baseline --gp-exact-cholesky
    python -m scripts.benchmark_station_scaling --model oat --device cpu --n-obs 64 128  # smoke test
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

import gpytorch
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from oat.model_module.model import OAT, _search_coords
from oat.predictor_module.gp_predictor import setup_gp_predictor
from oat.predictor_module.oat_predictor import setup_oat_cfg

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmarking" / "scaling_benchmark"

# Rough LV95 bounding box of Switzerland, used to place the fake stations.
EASTING_MIN, EASTING_MAX = 2_485_000.0, 2_834_000.0
NORTHING_MIN, NORTHING_MAX = 1_075_000.0, 1_296_000.0

DEFAULT_N_OBS = [2**p for p in range(6, 17)]  # 64 ... 65536

# gpytorch switches from exact Cholesky to iterative CG/Lanczos once a kernel
# matrix is larger than max_cholesky_size (default 800). Training uses that
# default, so we keep it unless --gp-exact-cholesky is given, which raises the
# threshold so the GP always does the exact O(N^3) solve.
GP_DEFAULT_MAX_CHOLESKY_SIZE = 800
GP_EXACT_MAX_CHOLESKY_SIZE = 10**9


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def load_cfg(
    model_name: str,
    config_path: Path | None = None,
    dataset_config_path: Path | None = None,
):
    if config_path is None:
        # No saved config given: compose the current default Hydra configs.
        overrides = [f"model={model_name}"]
        if model_name == "gp_baseline":
            overrides.append("predictor=gp_predictor")
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
            cfg = compose(config_name="config", overrides=overrides)
    else:
        # The config.yaml saved next to a training checkpoint, i.e. exactly the
        # model of that run (it is already resolved).
        cfg = OmegaConf.load(config_path)

    if dataset_config_path is not None:
        # Take the dataset part (input and target variables) from another run,
        # so that all benchmarked models see the same inputs and predict the
        # same variables. No data is loaded: the dataset part only sets the
        # shapes of the fake inputs. The output head must then predict those
        # variables too (setup_oat_cfg checks this).
        cfg.dataset = OmegaConf.load(dataset_config_path).dataset
        if "output_head" in cfg.model:
            cfg.model.output_head.variable_to_distribution = (
                cfg.dataset.obs_variable_to_distribution
            )
    return cfg


def build_model(model_name: str, cfg) -> torch.nn.Module:
    field_extent = {
        "min": [EASTING_MIN, NORTHING_MIN],
        "max": [EASTING_MAX, NORTHING_MAX],
    }
    if model_name == "oat":
        oat_cfg = setup_oat_cfg(cfg, field_extent, logger)
        return OAT(oat_cfg)
    if model_name == "gp_baseline":
        # The predictor only stores the transforms; the fake inputs are already
        # "normalized", so no transforms are needed.
        predictor = setup_gp_predictor(
            cfg=cfg,
            field_extent=field_extent,
            obs_vars=list(cfg.dataset.obs_vars),
            transforms=(None, None),
        )
        return predictor.model
    raise ValueError(f"Unknown model {model_name!r}, expected 'oat' or 'gp_baseline'")


def load_gp_weights(model: torch.nn.Module, checkpoint_path: Path) -> None:
    """Load trained GP weights from a GPPredictor checkpoint.

    Why: the approximate GP (CG/Lanczos above max_cholesky_size) needs as many
    iterations as the kernel matrix is ill-conditioned, which depends on the
    trained noise and lengthscales. With the initial values (noise 1.0) the
    matrix is well-conditioned and CG converges unrealistically fast. The exact
    Cholesky costs the same whatever the weights.

    The GP is built from the same config.yaml as the checkpoint, so all shapes
    match. The checkpoint holds the whole GPPredictor: the GP under the keys
    'model.*', next to the fitted input/output transforms, which the benchmark
    does not use. Only the GP part is loaded, with strict key matching.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    gp_state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(gp_state)


# --------------------------------------------------------------------------- #
# Fake inputs
# --------------------------------------------------------------------------- #
def make_fake_points(
    batch_size: int,
    n_points: int,
    n_nwp_channels: int,
    n_topo_channels: int,
    n_obs_vars: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """One group of points (stations or queries) as the model expects it."""
    easting = torch.empty(batch_size, n_points).uniform_(EASTING_MIN, EASTING_MAX)
    northing = torch.empty(batch_size, n_points).uniform_(NORTHING_MIN, NORTHING_MAX)
    elevation = torch.randn(batch_size, n_points)

    # Raw LV95 coordinates: used by OAT for the neighbour search.
    coords = torch.stack([easting, northing], dim=-1)

    # Geo-bias features (easting, northing, dem) are standardized by the input
    # transform in the real pipeline, so we standardize them here as well.
    easting_std = (easting - easting.mean()) / easting.std()
    northing_std = (northing - northing.mean()) / northing.std()
    geo = torch.stack([easting_std, northing_std, elevation], dim=-1)

    points = {
        "coords": coords,
        "geo": geo,
        "topo": torch.randn(batch_size, n_points, n_topo_channels),
        "global_tokens": torch.randn(batch_size, n_points, n_nwp_channels),
        # Non-negative, so the values are inside the support of every output
        # distribution (e.g. a lognormal for precipitation).
        "obs": torch.rand(batch_size, n_points, n_obs_vars),
        "mask_obs": torch.ones(batch_size, n_points, n_obs_vars),
    }
    return {key: value.to(device) for key, value in points.items()}


def make_fake_batch(
    cfg, batch_size: int, n_obs: int, n_query: int, device: torch.device
) -> dict[str, dict[str, torch.Tensor]]:
    global_field_name = cfg.dataset.roles["global"]
    n_nwp_channels = len(cfg.dataset.fields[global_field_name].inputs)
    n_topo_channels = len(cfg.dataset.topo_vars)
    n_obs_vars = len(cfg.dataset.obs_vars)

    observation_data = make_fake_points(
        batch_size, n_obs, n_nwp_channels, n_topo_channels, n_obs_vars, device
    )
    query_data = make_fake_points(
        batch_size, n_query, n_nwp_channels, n_topo_channels, n_obs_vars, device
    )
    # The predictor pops the query targets before calling the model.
    target_obs = query_data.pop("obs")
    target_mask_obs = query_data.pop("mask_obs")

    # sin/cos of a random time of day and day of year.
    day_angle = 2 * math.pi * torch.rand(batch_size)
    year_angle = 2 * math.pi * torch.rand(batch_size)
    time_emb = torch.stack(
        [day_angle.sin(), day_angle.cos(), year_angle.sin(), year_angle.cos()], dim=-1
    )

    return {
        "global_data": {"time_emb": time_emb.to(device)},
        "observation_data": observation_data,
        "query_data": query_data,
        "targets": {"target_obs": target_obs, "target_mask_obs": target_mask_obs},
    }


# --------------------------------------------------------------------------- #
# The two things we time
# --------------------------------------------------------------------------- #
def train_step(
    model_name: str, model: torch.nn.Module, batch: dict, gp_max_cholesky_size: int
) -> None:
    model.train()
    model.zero_grad(set_to_none=True)
    if model_name == "oat":
        # OATPredictor always asks for the pre-OAT output, so we do too.
        _, _, loss = model.forward_with_loss(
            global_data=batch["global_data"],
            observation_data=batch["observation_data"],
            query_data=batch["query_data"],
            target_obs=batch["targets"]["target_obs"],
            target_mask_obs=batch["targets"]["target_mask_obs"],
            return_pre_oat_output=True,
        )
        loss.backward()
    else:
        # The backward pass also goes through the Cholesky / CG solve, so it
        # must run inside the same gpytorch setting as the forward pass.
        with gpytorch.settings.max_cholesky_size(gp_max_cholesky_size):
            _, loss = model.forward_with_loss(
                global_data=batch["global_data"],
                observation_data=batch["observation_data"],
                query_data=batch["query_data"],
                target_obs=batch["targets"]["target_obs"],
                target_mask_obs=batch["targets"]["target_mask_obs"],
            )
            loss.backward()


def inference_step(
    model_name: str, model: torch.nn.Module, batch: dict, gp_max_cholesky_size: int
) -> None:
    model.eval()
    with torch.no_grad():
        if model_name == "oat":
            model.forward(
                global_data=batch["global_data"],
                observation_data=batch["observation_data"],
                query_data=batch["query_data"],
                return_pre_oat_output=True,
            )
        else:
            with gpytorch.settings.max_cholesky_size(gp_max_cholesky_size):
                model.forward(
                    global_data=batch["global_data"],
                    observation_data=batch["observation_data"],
                    query_data=batch["query_data"],
                )


def knn_step(model: OAT, batch: dict) -> None:
    """Only the neighbour searches of OAT._forward (step 5), same calls."""
    observation_coords = _search_coords(batch["observation_data"])
    query_coords = _search_coords(batch["query_data"])
    with torch.no_grad():
        # Stations <-> stations, for the encoder self-attention.
        model.mask_generator(coords_1=observation_coords)
        # Queries <-> queries, only used by decoder self-attention.
        if model.config.transformer.decoder_self_attention:
            model.mask_generator(coords_1=query_coords)
        # Stations -> queries, for the decoder cross-attention.
        model.mask_generator(coords_1=observation_coords, coords_2=query_coords)


def measure(step_fn, device: torch.device, n_warmup: int, n_repeat: int) -> dict:
    for _ in range(n_warmup):
        step_fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    times_s = []
    for _ in range(n_repeat):
        start = time.perf_counter()
        step_fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times_s.append(time.perf_counter() - start)

    if device.type == "cuda":
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1e9
    else:
        peak_memory_gb = float("nan")
    return {
        "median_s": statistics.median(times_s),
        "mean_s": statistics.mean(times_s),
        "min_s": min(times_s),
        "peak_memory_gb": peak_memory_gb,
    }


def free_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", required=True, choices=["oat", "gp_baseline"])
    parser.add_argument("--n-obs", type=int, nargs="+", default=DEFAULT_N_OBS)
    parser.add_argument(
        "--query-mode",
        choices=["fixed", "proportional"],
        default="fixed",
        help="fixed: N_query = --n-query. proportional: N_query = N_obs / 4 "
        "(the 80/20 observed/query split used in training).",
    )
    parser.add_argument("--n-query", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-repeat", type=int, default=10)
    parser.add_argument(
        "--gp-exact-cholesky",
        action="store_true",
        help="Force the exact Cholesky solve for every N_obs (gpytorch "
        "otherwise uses CG/Lanczos above 800 points, as in training).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config.yaml saved next to a training checkpoint: builds exactly "
        "that model. Without it, the default Hydra configs are composed.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=None,
        help="config.yaml whose dataset part (input and target variables) "
        "replaces the one of --config, so that all models get the same inputs "
        "and outputs.",
    )
    parser.add_argument(
        "--override",
        nargs="+",
        default=[],
        help="Config values to change after loading, as key=value, e.g. "
        "model.masking.mode=full for full attention.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="GP only: .ckpt whose trained weights are loaded, so that the "
        "approximate (CG) GP sees the trained conditioning. Must belong to the "
        "same run as --config.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Name of the run folder in place of --model, e.g. gp_oat or "
        "deep_gp_oat (both are --model gp_baseline).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    if args.gp_exact_cholesky:
        gp_max_cholesky_size = GP_EXACT_MAX_CHOLESKY_SIZE
        cholesky_tag = "_exact"
    else:
        gp_max_cholesky_size = GP_DEFAULT_MAX_CHOLESKY_SIZE
        cholesky_tag = ""

    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = load_cfg(args.model, args.config, args.dataset_config)
    if len(args.override) > 0:
        # e.g. --override model.masking.mode=full
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))

    # Same floating-point precision as in training (the GPs train in float64,
    # OAT in float32). Lightning's precision=64 creates the model, the batch and
    # every tensor made inside forward in float64. Setting the default dtype
    # before the model and the fake batches are created does the same here.
    if int(cfg.trainer.precision) == 64:
        torch.set_default_dtype(torch.float64)

    model = build_model(args.model, cfg)
    if args.checkpoint is not None:
        if args.model != "gp_baseline":
            # The benchmarked OAT predicts a different set of variables than the
            # paper checkpoint (see --dataset-config), so its shapes differ.
            # OAT's cost does not depend on its weights anyway.
            raise ValueError("--checkpoint is only supported for --model gp_baseline")
        load_gp_weights(model, args.checkpoint)
    model = model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"Built {args.model} with {n_parameters} parameters on {device}")

    # One folder per run, e.g.
    #   benchmarking/scaling_benchmark/2026-10-05_16-30-00_oat_fixed_B1/
    #       results.csv           timings and peak memory
    #       model_config.yaml     the Hydra model config the model was built from
    #       dataset_config.yaml   the Hydra dataset config (sets the channel counts)
    #       benchmark_args.yaml   the command-line arguments of this run
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.label is None:
        label = args.model
    else:
        label = args.label
    run_name = (
        f"{run_timestamp}_{label}_{args.query_mode}"
        f"_B{args.batch_size}{cholesky_tag}"
    )
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    results_path = run_dir / "results.csv"

    (run_dir / "model_config.yaml").write_text(OmegaConf.to_yaml(cfg.model, resolve=True))
    (run_dir / "dataset_config.yaml").write_text(
        OmegaConf.to_yaml(cfg.dataset, resolve=True)
    )
    benchmark_args = {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()}
    benchmark_args["gp_max_cholesky_size"] = gp_max_cholesky_size
    benchmark_args["dtype"] = str(torch.get_default_dtype())
    benchmark_args["n_parameters"] = n_parameters
    if device.type == "cuda":
        benchmark_args["device_name"] = torch.cuda.get_device_name(device)
    else:
        benchmark_args["device_name"] = "cpu"
    (run_dir / "benchmark_args.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.create(benchmark_args))
    )
    logger.info(f"Writing results to {run_dir}")

    step_names = ["train", "inference"]
    # Time the neighbour search on its own only if the model does one: with
    # full attention (masking.mode=full) OAT has no neighbour search.
    if args.model == "oat" and cfg.model.masking.mode != "full":
        step_names.append("knn")

    columns = [
        "model", "step", "n_obs", "n_query", "batch_size",
        "median_s", "mean_s", "min_s", "peak_memory_gb", "status",
    ]
    with open(results_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()

        # Once a step runs out of memory, larger N_obs will too: skip them.
        out_of_memory = {step_name: False for step_name in step_names}

        for n_obs in sorted(args.n_obs):
            if args.query_mode == "fixed":
                n_query = args.n_query
            else:
                n_query = max(1, n_obs // 4)

            for step_name in step_names:
                row = {
                    "model": args.model,
                    "step": step_name,
                    "n_obs": n_obs,
                    "n_query": n_query,
                    "batch_size": args.batch_size,
                    "median_s": float("nan"),
                    "mean_s": float("nan"),
                    "min_s": float("nan"),
                    "peak_memory_gb": float("nan"),
                    "status": "ok",
                }
                if out_of_memory[step_name]:
                    row["status"] = "skipped_after_oom"
                    writer.writerow(row)
                    continue

                free_memory(device)
                batch = make_fake_batch(cfg, args.batch_size, n_obs, n_query, device)

                if step_name == "train":
                    step_fn = lambda: train_step(
                        args.model, model, batch, gp_max_cholesky_size
                    )
                elif step_name == "inference":
                    step_fn = lambda: inference_step(
                        args.model, model, batch, gp_max_cholesky_size
                    )
                else:
                    step_fn = lambda: knn_step(model, batch)

                try:
                    row.update(measure(step_fn, device, args.n_warmup, args.n_repeat))
                except torch.cuda.OutOfMemoryError:
                    row["status"] = "oom"
                    out_of_memory[step_name] = True
                except Exception as error:  # e.g. NotPSDError from the GP Cholesky
                    row["status"] = f"error: {type(error).__name__}: {error}"[:200]

                del batch, step_fn
                writer.writerow(row)
                csv_file.flush()
                logger.info(
                    f"{args.model:12s} {step_name:9s} N_obs={n_obs:6d} N_q={n_query:6d} "
                    f"median={row['median_s']:.4f}s peak={row['peak_memory_gb']:.2f}GB "
                    f"{row['status']}"
                )

    logger.info(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
