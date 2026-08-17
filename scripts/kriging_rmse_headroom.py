"""LOSO kriging-RMSE headroom check.

Decides whether the mean/gate channel is worth building, or whether we just stick with the
current model. Question: does fixed-kernel residual kriging of neighbouring station observations
reduce held-out-station RMSE *on top of* the transformer mean? If not, the transformer already
captures the exploitable spatial structure -> no headroom.

Deployment geometry (targets=test, context=train+val). One model forward per batch gives, per
variable: the OAT mean mu_T and the background (pre-OAT) mean mu_b at both context and test points.
Background innovations d = y_ctx - mu_b(ctx) are kriged to each test station with a fixed RBF kernel
(small (ell, sigma_r^2) grid). We compare RMSE of: mu_T, mu_b, mu_b+c (pure kriging), mu_T+c
(undamped residual = headroom upper bound), mu_T+K*c (K-damped, the note's design) -- overall and
stratified by distance-to-nearest-valid-context bin. No training.

Usage:
    python -m scripts.kriging_rmse_headroom --checkpoint <ckpt> --time-stride 6
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.spatial import cKDTree

from oat.data_module.legacy.data_split import EvalDataset, create_splits
from oat.data_module.setup import setup_data_and_transform
from oat.predictor_module.checkpoint_utils import load_checkpoint
from scripts.power_check_loso import latlon_to_xyz, haversine_km
from scripts.calibration_diagnostic_loso import to_device, bin_of, bin_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("kriging_rmse_headroom")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--distance-bins", default="2,5,10")
    ap.add_argument("--time-stride", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--k", type=int, default=16, help="Neighbours per test station.")
    ap.add_argument("--length-scales-km", default="5,10,20,50")
    ap.add_argument("--sigma_r2", default="0.05,0.2,1.0", help="Obs-noise variances (normalized units).")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="data/results")
    args = ap.parse_args()

    edges = [float(x) for x in args.distance_bins.split(",")]
    labels = bin_labels(edges)
    length_scales = [float(x) for x in args.length_scales_km.split(",")]
    sigma_r2s = [float(x) for x in args.sigma_r2.split(",")]
    device = torch.device(args.device)
    k = args.k

    predictor, cfg = load_checkpoint(None, args.checkpoint, map_location="cpu")
    quiet = logging.getLogger("khr.quiet")
    quiet.setLevel(logging.WARNING)
    dataset, _, _, _, _, input_transform, output_transform = setup_data_and_transform(cfg, quiet)
    if hasattr(predictor, "set_index_projection"):
        predictor.set_index_projection(dataset)
    predictor.input_transform = input_transform
    predictor.output_transform = output_transform
    predictor = predictor.to(device).eval()
    torch.set_grad_enabled(False)  # inference + fixed-kernel kriging only; no graph ever needed
    obs_vars = list(cfg.dataset.obs_vars)

    splits = create_splits(dataset, **cfg.split, logger=quiet)
    train_idx, val_idx, test_idx = splits["station_indices"]
    strided_times = np.asarray(splits["test_time_indices"], dtype=int)[:: args.time_stride]
    target_idx = np.asarray(test_idx, dtype=int)
    context_idx = np.asarray(list(train_idx) + list(val_idx), dtype=int)
    n_ctx, n_tgt = context_idx.size, target_idx.size
    logger.info("targets=%d context=%d hours=%d k=%d", n_tgt, n_ctx, strided_times.size, k)

    coords = dataset.station_coords.numpy()
    tgt_lon, tgt_lat = coords[target_idx, 0], coords[target_idx, 1]
    ctx_lon, ctx_lat = coords[context_idx, 0], coords[context_idx, 1]

    # Static k-NN of each test station among context stations, + great-circle distances.
    _, nn = cKDTree(latlon_to_xyz(ctx_lat, ctx_lon)).query(latlon_to_xyz(tgt_lat, tgt_lon), k=k)  # [n_tgt,k]
    nb_lat, nb_lon = ctx_lat[nn], ctx_lon[nn]  # [n_tgt,k]
    d_qS = haversine_km(tgt_lat[:, None], tgt_lon[:, None], nb_lat, nb_lon)  # [n_tgt,k]
    D_SS = haversine_km(nb_lat[:, :, None], nb_lon[:, :, None], nb_lat[:, None, :], nb_lon[:, None, :])  # [n_tgt,k,k]
    nn_t = torch.as_tensor(nn, device=device)
    d_qS_t = torch.as_tensor(d_qS, dtype=torch.float32, device=device)
    D_SS_t = torch.as_tensor(D_SS, dtype=torch.float32, device=device)

    # Precompute RBF kernels per length scale (static geometry).
    kern = {
        ell: (torch.exp(-(d_qS_t ** 2) / (2 * ell ** 2)),        # k_qS [n_tgt,k]
              torch.exp(-(D_SS_t ** 2) / (2 * ell ** 2)))         # K_SS [n_tgt,k,k]
        for ell in length_scales
    }

    eval_idx = np.concatenate([context_idx, target_idx])
    eval_ds = EvalDataset(dataset, eval_indices=eval_idx, observed_indices=context_idx, time_indices=strided_times)
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    n_bins = len(labels)
    configs = [(ell, sr2) for ell in length_scales for sr2 in sigma_r2s]
    # Accumulators: sum of squared error and count, per variable, per bin.
    cnt = {v: np.zeros(n_bins) for v in obs_vars}
    sse_base = {v: {"muT": np.zeros(n_bins), "mub": np.zeros(n_bins)} for v in obs_vars}
    sse_cfg = {v: {c: {"mub_c": np.zeros(n_bins), "muT_c": np.zeros(n_bins), "muT_Kc": np.zeros(n_bins)}
                   for c in configs} for v in obs_vars}

    def accum(sse_by_bin, err2, bid, valid):
        for b in range(n_bins):
            m = valid & (bid == b)
            if m.any():
                sse_by_bin[b] += float(err2[m].sum())

    for bi, batch in enumerate(loader):
        if args.max_batches is not None and bi >= args.max_batches:
            break
        batch = to_device(batch, device)
        norm = predictor._normalize_batch(batch)
        q_obs = norm["query_data"]["obs"].detach().clone()          # [B, n_ctx+n_tgt, V] normalized
        q_mask = norm["query_data"]["mask_obs"].detach().clone().bool()
        with torch.no_grad():
            out = predictor(norm, return_loss_and_targets=False, n_samples=1)
        B = q_obs.shape[0]

        for v, var in enumerate(obs_vars):
            mu_T_all = out[var].mean                       # [B, n_ctx+n_tgt]
            mu_b_all = out["pre_oat_output"][var].mean
            obs_ctx = q_obs[:, :n_ctx, v]
            valid_ctx = q_mask[:, :n_ctx, v]
            d_full = torch.where(valid_ctx, obs_ctx - mu_b_all[:, :n_ctx], torch.zeros_like(obs_ctx))  # [B,n_ctx]

            mu_T = mu_T_all[:, n_ctx:]                      # [B,n_tgt]
            mu_b = mu_b_all[:, n_ctx:]
            y = q_obs[:, n_ctx:, v]
            valid_t = q_mask[:, n_ctx:, v].cpu().numpy()    # [B,n_tgt]

            d_nb = d_full[:, nn_t]                          # [B,n_tgt,k]
            valid_nb = valid_ctx[:, nn_t]                   # [B,n_tgt,k]
            # nearest valid context distance per (hour, test) from the static-k neighbours
            dist_nb = d_qS_t.unsqueeze(0).expand(B, -1, -1).clone()
            dist_nb[~valid_nb] = float("inf")
            nearest = dist_nb.min(dim=-1).values.cpu().numpy()  # [B,n_tgt]
            bid = bin_of(nearest, edges)

            err_muT = ((mu_T - y) ** 2).cpu().numpy()
            err_mub = ((mu_b - y) ** 2).cpu().numpy()
            accum(sse_base[var]["muT"], err_muT, bid, valid_t)
            accum(sse_base[var]["mub"], err_mub, bid, valid_t)
            cnt_flat = np.zeros(n_bins)
            for b in range(n_bins):
                cnt_flat[b] = int((valid_t & (bid == b)).sum())
            cnt[var] += cnt_flat

            for (ell, sr2) in configs:
                k_qS, K_SS = kern[ell]                       # [n_tgt,k], [n_tgt,k,k]
                R = sr2 + (~valid_nb).float() * 1e6          # [B,n_tgt,k]
                A = K_SS.unsqueeze(0) + torch.diag_embed(R)  # [B,n_tgt,k,k]
                kqS_b = k_qS.unsqueeze(0).expand(B, -1, -1)  # [B,n_tgt,k]
                rhs = torch.stack([d_nb, kqS_b], dim=-1)     # [B,n_tgt,k,2]
                sol = torch.linalg.solve(A, rhs)             # [B,n_tgt,k,2]
                c = (kqS_b * sol[..., 0]).sum(-1)            # [B,n_tgt]
                Kgain = (kqS_b * sol[..., 1]).sum(-1).clamp(0.0, 1.0)  # kappa(q,q)=1
                accum(sse_cfg[var][(ell, sr2)]["mub_c"], ((mu_b + c - y) ** 2).cpu().numpy(), bid, valid_t)
                accum(sse_cfg[var][(ell, sr2)]["muT_c"], ((mu_T + c - y) ** 2).cpu().numpy(), bid, valid_t)
                accum(sse_cfg[var][(ell, sr2)]["muT_Kc"], ((mu_T + Kgain * c - y) ** 2).cpu().numpy(), bid, valid_t)

        if bi % 10 == 0:
            logger.info("batch %d", bi)

    # ── Reduce to RMSE and report (normalized units). ──
    def rmse(sse_by_bin, sel_bins):
        s = sum(sse_by_bin[b] for b in sel_bins)
        n = sum(cnt[var][b] for b in sel_bins)
        return float(np.sqrt(s / n)) if n > 0 else float("nan")

    near_bins = [b for b, lab in enumerate(labels) if lab in (labels[0], labels[1])]  # <2 and 2-5 km
    all_bins = list(range(n_bins))
    results = {"checkpoint": str(args.checkpoint), "geometry": "deployment", "time_stride": args.time_stride,
               "k": k, "units": "normalized", "distance_bins_km": edges, "per_variable": {}}

    for var in obs_vars:
        rmse_muT_all = rmse(sse_base[var]["muT"], all_bins)
        rmse_muT_near = rmse(sse_base[var]["muT"], near_bins)
        rmse_mub_all = rmse(sse_base[var]["mub"], all_bins)
        # Pick the best config by near-bin muT_c RMSE (headroom concentrates near stations).
        best = min(configs, key=lambda c: rmse(sse_cfg[var][c]["muT_c"], near_bins))
        def pack(est):
            return {"all": rmse(sse_cfg[var][best][est], all_bins),
                    "near<5km": rmse(sse_cfg[var][best][est], near_bins),
                    **{labels[b]: rmse(sse_cfg[var][best][est], [b]) for b in all_bins}}
        rv = {
            "n_per_bin": {labels[b]: int(cnt[var][b]) for b in all_bins},
            "best_config": {"length_scale_km": best[0], "sigma_r2": best[1]},
            "rmse_muT_current": {"all": rmse_muT_all, "near<5km": rmse_muT_near,
                                 **{labels[b]: rmse(sse_base[var]["muT"], [b]) for b in all_bins}},
            "rmse_mub_background": {"all": rmse_mub_all},
            "rmse_mub_plus_c_pure_kriging": pack("mub_c"),
            "rmse_muT_plus_c_residual": pack("muT_c"),
            "rmse_muT_plus_Kc_damped": pack("muT_Kc"),
        }
        # Headroom = relative RMSE reduction of the best residual variant vs current model.
        rv["pct_delta_muT_plus_c"] = {
            "all": 100.0 * (rv["rmse_muT_plus_c_residual"]["all"] - rmse_muT_all) / rmse_muT_all,
            "near<5km": 100.0 * (rv["rmse_muT_plus_c_residual"]["near<5km"] - rmse_muT_near) / rmse_muT_near,
        }
        results["per_variable"][var] = rv

        logger.info("── %s ── best ell=%gkm sigma_r2=%g", var, best[0], best[1])
        logger.info("  RMSE current mu_T:    all=%.4f  near<5km=%.4f", rmse_muT_all, rmse_muT_near)
        logger.info("  RMSE background mu_b:  all=%.4f", rmse_mub_all)
        logger.info("  RMSE mu_b + c (krig):  all=%.4f  near=%.4f", rv["rmse_mub_plus_c_pure_kriging"]["all"], rv["rmse_mub_plus_c_pure_kriging"]["near<5km"])
        logger.info("  RMSE mu_T + c:         all=%.4f  near=%.4f  (Δ%% all=%+.2f near=%+.2f)",
                    rv["rmse_muT_plus_c_residual"]["all"], rv["rmse_muT_plus_c_residual"]["near<5km"],
                    rv["pct_delta_muT_plus_c"]["all"], rv["pct_delta_muT_plus_c"]["near<5km"])
        logger.info("  RMSE mu_T + K·c:       all=%.4f  near=%.4f",
                    rv["rmse_muT_plus_Kc_damped"]["all"], rv["rmse_muT_plus_Kc_damped"]["near<5km"])
        verdict = ("HEADROOM near stations -> mean channel worth building"
                   if rv["pct_delta_muT_plus_c"]["near<5km"] < -2.0
                   else "no material headroom -> stick with the current model (for this variable)")
        rv["verdict"] = verdict
        logger.info("  VERDICT: %s", verdict)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kriging_rmse_headroom.json").write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", out_dir / "kriging_rmse_headroom.json")


if __name__ == "__main__":
    main()
