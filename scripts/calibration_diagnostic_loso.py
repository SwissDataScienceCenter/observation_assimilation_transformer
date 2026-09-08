"""Near-station PIT/rank-histogram calibration diagnostic across a distance gradient.

This is the gate for the Kalman-gain *variance* channel (docs/kalman_gain_head_note.tex,
Section "Premise verification"). It answers: does the free-form predictive head become
over-dispersed *specifically near observations*?

Deployment geometry (matches how the checkpoint was trained/evaluated): targets = the test
stations, context = the train+val stations. For every scorable target-station x hour we
compute the predictive PIT value F_pred(y) (via the head's ``innovation`` = Phi^-1(F(y)),
which is ~N(0,1) under calibration) and bin it by the distance to the nearest *valid* context
station (<2, 2-5, 5-10, >=10 km). Reads the checkpoint's sibling config.yaml and loads weights.

Decision rule (GRILL REV 2):
  * Over-dispersion = center-heavy PIT (std of the N(0,1) innovation < 1) that *worsens*
    toward zero distance -> a confirmed target: build the variance channel.
  * If the <2 km bin is already calibrated (std ~ 1) -> the representativeness floor dominates,
    the correct gain is ~0, flat uncertainty is correct -> drop the variance channel, keep
    only the mean/gate work.

Usage:
    python -m scripts.calibration_diagnostic_loso \
        --checkpoint checkpoints/oat_L/.../best-model-...ckpt --time-stride 6
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm as _norm
from torch.utils.data import DataLoader

from oat.data_module.legacy.data_split import EvalDataset, create_splits
from oat.data_module.setup import setup_data_and_transform
from oat.predictor_module.checkpoint_utils import load_checkpoint
from scripts.power_check_loso import nearest_valid_distance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("calibration_diagnostic_loso")


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def compute_pit(dist, y: torch.Tensor, rng: np.random.Generator) -> np.ndarray:
    """Probability integral transform F(y), with a *randomized* PIT at a point mass.

    Under calibration the PIT is Uniform(0,1). For a purely continuous head (Normal,
    LogNormal) this is just F(y). For the zero-inflated head the mass at 0 makes the
    plain PIT discrete/biased, so at y==0 we draw u ~ Uniform(F(0^-), F(0)) = Uniform(0, gate);
    for y>0 the continuous part needs no randomization.
    """
    F = dist.cdf(y)
    if hasattr(dist, "gate"):  # zero-inflated: randomize across the mass at 0
        lower = torch.where(y == 0, torch.zeros_like(F), F)  # F(y^-)
        V = torch.as_tensor(rng.random(tuple(y.shape)), dtype=F.dtype, device=F.device)
        u = lower + V * (F - lower)
    else:
        u = F
    eps = torch.finfo(u.dtype).eps
    return u.clamp(eps, 1 - eps).detach().cpu().numpy()


def wet_continuous_pit(dist, y: torch.Tensor) -> np.ndarray:
    """Continuous-part PIT for the zero-inflated head, wet obs only.

    F(y | Y>0) = LogNormal.cdf(y); Uniform(0,1) iff the *continuous rain-rate spread*
    is calibrated. Isolates the log-normal dispersion from the (dominant) dry mass, so
    it is the quantity a variance channel would actually affect. Returns NaN for dry obs.
    """
    eps = torch.finfo(y.dtype).eps
    ln = torch.distributions.LogNormal(dist.loc, dist.scale)
    u = ln.cdf(y.clamp(min=eps))
    u = torch.where(y > 0, u.clamp(eps, 1 - eps), torch.full_like(u, float("nan")))
    return u.detach().cpu().numpy()


def bin_of(dist_km: np.ndarray, edges) -> np.ndarray:
    """Map distances to integer bin ids: 0=<edges[0], ..., len(edges)=>=edges[-1]/none."""
    out = np.full(dist_km.shape, len(edges), dtype=int)
    lo = 0.0
    for b, hi in enumerate(edges):
        out[(dist_km >= lo) & (dist_km < hi)] = b
        lo = hi
    return out


def bin_labels(edges):
    labels, lo = [], 0.0
    for hi in edges:
        labels.append(f"<{hi:g}km" if lo == 0.0 else f"{lo:g}-{hi:g}km")
        lo = hi
    labels.append(f">={edges[-1]:g}km/none")
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--distance-bins", default="2,5,10", help="Comma-separated bin edges in km.")
    ap.add_argument("--time-stride", type=int, default=6, help="Subsample test hours (power is ~1000x surplus).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-batches", type=int, default=None, help="Debug: cap number of batches.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="data/results")
    ap.add_argument("--fig-dir", default="data/figures")
    ap.add_argument("--seed", type=int, default=0, help="Seed for the randomized-PIT draws.")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    edges = [float(x) for x in args.distance_bins.split(",")]
    labels = bin_labels(edges)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    predictor, cfg = load_checkpoint(None, args.checkpoint, map_location="cpu")
    logger.info("Loaded checkpoint (dataset_type=%s, device=%s).", cfg.dataset.dataset_type, device)

    # Dataset + transforms exactly as training built them; reuse for consistency.
    # Feed setup/create_splits a quiet logger so the multi-thousand station-index dump
    # doesn't drown the diagnostic output.
    quiet = logging.getLogger("power_check_loso.quiet")
    quiet.setLevel(logging.WARNING)
    dataset, _, _, _, _, input_transform, output_transform = setup_data_and_transform(cfg, quiet)
    if hasattr(predictor, "set_index_projection"):
        predictor.set_index_projection(dataset)
    predictor.input_transform = input_transform
    predictor.output_transform = output_transform
    predictor = predictor.to(device).eval()

    obs_vars = list(cfg.dataset.obs_vars)

    splits = create_splits(dataset, **cfg.split, logger=quiet)
    train_idx, val_idx, test_idx = splits["station_indices"]
    test_time_indices = np.asarray(splits["test_time_indices"], dtype=int)
    strided_times = test_time_indices[:: args.time_stride]

    target_idx = np.asarray(test_idx, dtype=int)               # order == query_data order
    context_idx = np.asarray(list(train_idx) + list(val_idx), dtype=int)
    logger.info(
        "%d target stations, %d context stations, %d test hours (stride %d -> %d).",
        target_idx.size, context_idx.size, test_time_indices.size, args.time_stride, strided_times.size,
    )

    coords = dataset.station_coords.numpy()  # [N, 2] = (lon, lat)
    tgt_lon, tgt_lat = coords[target_idx, 0], coords[target_idx, 1]
    ctx_lon, ctx_lat = coords[context_idx, 0], coords[context_idx, 1]

    # Per-variable nearest-valid-context distance [T_strided, n_target], aligned to loader order.
    mask_strided = dataset.obs_mask_tensor[strided_times].numpy().astype(bool)  # [T, N, V]
    nearest = {
        var: nearest_valid_distance(
            tgt_lat, tgt_lon, ctx_lat, ctx_lon, mask_strided[:, context_idx, v], edges[-1]
        )
        for v, var in enumerate(obs_vars)
    }

    eval_ds = EvalDataset(dataset, eval_indices=target_idx, observed_indices=context_idx, time_indices=strided_times)
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Accumulate the PIT value u=F(y) (Uniform under calibration) and its distance, per variable.
    pit_acc = {var: [] for var in obs_vars}
    d_acc = {var: [] for var in obs_vars}
    # Wet-only continuous PIT for zero-inflated heads (isolates rain-rate spread from the dry mass).
    pit_wet_acc: dict[str, list] = {}
    d_wet_acc: dict[str, list] = {}

    row0 = 0
    for bi, batch in enumerate(loader):
        if args.max_batches is not None and bi >= args.max_batches:
            break
        batch = to_device(batch, device)
        norm = predictor._normalize_batch(batch)
        tgt_obs = norm["query_data"]["obs"].detach().clone()          # [B, S_q, V] (normalized)
        tgt_mask = norm["query_data"]["mask_obs"].detach().clone().bool()
        with torch.no_grad():
            dists = predictor(norm, return_loss_and_targets=False, n_samples=1)
        bsz = tgt_obs.shape[0]
        dist_rows = {var: nearest[var][row0 : row0 + bsz] for var in obs_vars}  # [B, n_target]
        row0 += bsz

        for v, var in enumerate(obs_vars):
            try:
                u = compute_pit(dists[var], tgt_obs[..., v], rng)  # [B, S_q] in (0,1)
            except Exception as exc:  # noqa: BLE001
                logger.warning("PIT failed for %s (%s); skipping this variable.", var, exc)
                continue
            valid = tgt_mask[..., v].cpu().numpy()
            m = valid & np.isfinite(u)
            pit_acc[var].append(u[m])
            d_acc[var].append(dist_rows[var][m])
            if hasattr(dists[var], "gate"):  # zero-inflated: also wet-only continuous PIT
                uw = wet_continuous_pit(dists[var], tgt_obs[..., v])
                mw = valid & np.isfinite(uw)
                pit_wet_acc.setdefault(var, []).append(uw[mw])
                d_wet_acc.setdefault(var, []).append(dist_rows[var][mw])

        if bi % 20 == 0:
            logger.info("batch %d (rows so far %d)", bi, row0)

    results = {
        "checkpoint": str(args.checkpoint),
        "geometry": "deployment (targets=test, context=train+val)",
        "time_stride": args.time_stride,
        "distance_bins_km": edges,
        "note": "z is the N(0,1) PIT-innovation; std<1 => over-dispersed, std>1 => under-dispersed.",
        "per_variable": {},
    }

    hist_bins = np.linspace(0.0, 1.0, 21)  # 20-bin PIT histogram

    def per_bin_stats(pit, d, hists_out=None):
        bid = bin_of(d, edges)
        z = _norm.ppf(pit)  # N(0,1) under calibration; std<1 over-dispersed, std>1 under-dispersed
        per_bin = {}
        for b, label in enumerate(labels):
            sel = bid == b
            n = int(sel.sum())
            if n == 0:
                per_bin[label] = {"n": 0}
                continue
            zb = z[sel]
            per_bin[label] = {
                "n": n,
                "z_mean": float(zb.mean()),
                "z_std": float(zb.std()),
                "pit_center_frac_0.25_0.75": float(((pit[sel] >= 0.25) & (pit[sel] <= 0.75)).mean()),
            }
            if hists_out is not None:
                hists_out[label] = np.histogram(pit[sel], bins=hist_bins)[0].tolist()
        return per_bin

    def log_report(title, per_bin):
        logger.info("── %s ── (z_std: 1.0=calibrated, <1=over-dispersed, >1=under-dispersed/overconfident)", title)
        for label in labels:
            s = per_bin[label]
            if s.get("n"):
                logger.info("  %-14s n=%-8d z_std=%.3f z_mean=%+.3f center[.25,.75]=%.3f",
                            label, s["n"], s["z_std"], s["z_mean"], s["pit_center_frac_0.25_0.75"])

    pit_hists = {}
    for var in obs_vars:
        if not pit_acc[var]:
            continue
        pit_hists[var] = {}
        per_bin = per_bin_stats(np.concatenate(pit_acc[var]), np.concatenate(d_acc[var]), pit_hists[var])
        results["per_variable"][var] = per_bin
        log_report(var, per_bin)

    results["per_variable_wet_continuous"] = {}
    for var in pit_wet_acc:
        per_bin = per_bin_stats(np.concatenate(pit_wet_acc[var]), np.concatenate(d_wet_acc[var]))
        results["per_variable_wet_continuous"][var] = per_bin
        log_report(f"{var} [wet-only continuous spread]", per_bin)

    # Three-way verdict on the primary continuous head (temp).
    if "temp" in results["per_variable"]:
        near = results["per_variable"]["temp"].get(labels[0], {})
        far = results["per_variable"]["temp"].get(labels[-1], {})
        if near.get("n") and far.get("n"):
            ns, fs = near["z_std"], far["z_std"]
            if ns < 0.9 and ns < fs - 0.02:
                verdict = ("OVER-DISPERSED near stations (uncertainty too wide) -> the note's hypothesis; "
                           "BUILD the variance channel (Kalman gain shrinks sigma near obs).")
            elif ns > 1.1 and ns > fs + 0.02:
                verdict = ("UNDER-DISPERSED near stations (overconfident; sigma too small vs representativeness "
                           "error) -> a variance-SHRINKING Kalman gain would make this WORSE. Do NOT build the "
                           "variance channel; keep only the mean/gate work.")
            else:
                verdict = ("near bin ~calibrated -> no near-station variance problem; drop the variance channel, "
                           "keep mean/gate.")
            results["temp_verdict"] = verdict
            logger.info("VERDICT (temp): %s  [near z_std=%.3f, far z_std=%.3f]", verdict, ns, fs)

    out_dir, fig_dir = Path(args.out_dir), Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_diagnostic_loso.json").write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", out_dir / "calibration_diagnostic_loso.json")

    if not args.no_plot and pit_hists:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        centers = (hist_bins[:-1] + hist_bins[1:]) / 2
        vars_present = list(pit_hists.keys())
        fig, axes = plt.subplots(len(vars_present), len(labels), figsize=(3 * len(labels), 3 * len(vars_present)), squeeze=False)
        for r, var in enumerate(vars_present):
            for c, label in enumerate(labels):
                ax = axes[r][c]
                h = pit_hists[var].get(label)
                if h:
                    ax.bar(centers, np.asarray(h) / max(1, sum(h)), width=0.045)
                ax.axhline(1.0 / (len(hist_bins) - 1), color="k", ls="--", lw=0.8)
                ax.set_title(f"{var} {label}", fontsize=8)
                ax.set_ylim(0, None)
        fig.suptitle("Near-station PIT histograms (flat=calibrated, humped=over-dispersed)")
        fig.tight_layout()
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_dir / "calibration_diagnostic_loso_pit.png", dpi=120)
        logger.info("Wrote %s", fig_dir / "calibration_diagnostic_loso_pit.png")


if __name__ == "__main__":
    main()
