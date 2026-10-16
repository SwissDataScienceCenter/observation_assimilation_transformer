"""
GPU memory profiler for a single forward pass (no DDP, no W&B).

Usage (same Hydra overrides as train_model):
    poetry run python -m scripts.profile_memory \
        dataset=era5_meteostat/full \
        dataloader.train.batch_size=32 \
        dataloader.train.num_workers=2 \
        model.d_model=32

Prints per-stage memory and a top-30 live-tensor breakdown at the end.
"""

from __future__ import annotations

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from oat.data_module.field import Field
from oat.data_module.setup import setup_data_and_transform
from oat.predictor_module.oat_predictor import setup_oat_cfg, setup_oat_predictor

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cuda", 0)
_STAGE = 0


def mem(tag: str) -> None:
    global _STAGE
    alloc  = torch.cuda.memory_allocated(_DEVICE) / 1e9
    reserv = torch.cuda.memory_reserved(_DEVICE)  / 1e9
    peak   = torch.cuda.max_memory_allocated(_DEVICE) / 1e9
    print(f"[{_STAGE:02d}] {tag:<60s}  alloc={alloc:6.2f} GB  reserv={reserv:6.2f} GB  peak={peak:6.2f} GB")
    _STAGE += 1

@hydra.main(config_path="../configs/", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    torch.manual_seed(cfg.seed)

    # ── data ──────────────────────────────────────────────────────────────
    dataset, len_train, train_loader, val_loader, _, input_transform, output_transform = \
        setup_data_and_transform(cfg, logger)

    # ── model (single GPU, no DDP) ────────────────────────────────────────
    from oat.predictor_module.field_batch_utils import resolve_index_projection

    oat_model_cfg = setup_oat_cfg(
        cfg,
        logger,
        project_index_coords=resolve_index_projection(dataset),
    )
    predictor = setup_oat_predictor(
        model_cfg=oat_model_cfg,
        estimated_total_step=1,
        var_names=list(cfg.dataset.obs_vars),
        transforms=[input_transform, output_transform],
        cfg=cfg,
    )
    predictor.set_index_projection(dataset)
    predictor = predictor.to(_DEVICE).eval()
    mem("model on GPU")

    # ── one validation batch ──────────────────────────────────────────────
    batch = next(iter(val_loader))
    mem("after fetching batch (CPU)")

    # Field-aware transfer (plain tensor.to() does not know about Field
    # dataclasses, so it silently leaves field-collated data on CPU).
    batch = predictor.transfer_batch_to_device(batch, _DEVICE, 0)
    mem("batch on GPU")

    print("\n── Batch tensor sizes ──────────────────────────────────────────────────")
    def _report(obj, prefix=""):
        if isinstance(obj, torch.Tensor):
            print(f"  {prefix:<60s} {str(tuple(obj.shape)):<30s} {obj.nbytes/1e6:8.1f} MB")
        elif isinstance(obj, Field):
            _report(obj.coords, f"{prefix}.coords")
            _report(obj.values, f"{prefix}.values")
            if obj.mask is not None:
                _report(obj.mask, f"{prefix}.mask")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _report(v, f"{prefix}.{k}" if prefix else k)
    _report(batch)
    print()

    # Predictor-owned normalization: applies input_transform and unwraps
    # Field batches into the flat tensors forward() expects (plain
    # input_transform(batch) alone leaves Field objects in place).
    batch = predictor._normalize_batch(batch)
    mem("after normalize (transform + field unwrap)")

    # ── forward pass with memory snapshot ────────────────────────────────
    torch.cuda.reset_peak_memory_stats(_DEVICE)
    torch.cuda.memory._record_memory_history(max_entries=200_000)

    with torch.no_grad():
        try:
            _ = predictor(batch, return_loss_and_targets=False)
            mem("forward pass complete (no OOM)")
        except torch.cuda.OutOfMemoryError as e:
            mem("*** OOM during forward ***")
            print(f"\nOOM error: {e}\n")

    # ── snapshot analysis ─────────────────────────────────────────────────
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)

    snap_path = "/tmp/oat_mem_snapshot.pickle"
    import pickle
    with open(snap_path, "wb") as f:
        pickle.dump(snap, f)
    print(f"\nFull snapshot saved to {snap_path}")

    print("\n── Top-30 live tensors by size ─────────────────────────────────────────")
    blocks = []
    for seg in snap.get("segments", []):
        for blk in seg.get("blocks", []):
            if blk.get("state") == "active_allocated":
                frames = blk.get("frames", [])
                src = next(
                    (f for f in reversed(frames) if "site-packages" not in f.get("filename", "")),
                    frames[-1] if frames else {},
                )
                blocks.append((blk.get("size", 0), src))
    blocks.sort(key=lambda x: x[0], reverse=True)
    for size, src in blocks[:30]:
        print(f"  {size/1e6:8.1f} MB  {src.get('filename','?')}:{src.get('line','?')}  {src.get('name','')}")

    print("\n── torch.cuda.memory_summary ───────────────────────────────────────────")
    print(torch.cuda.memory_summary(_DEVICE))


if __name__ == "__main__":
    main()
