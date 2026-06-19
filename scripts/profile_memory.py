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


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


@hydra.main(config_path="../configs/", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    torch.manual_seed(cfg.seed)

    # ── data ──────────────────────────────────────────────────────────────
    dataset, len_train, train_loader, val_loader, _, input_transform, output_transform = \
        setup_data_and_transform(cfg, logger)

    field_extent = {k: v.tolist() for k, v in dataset.nwp_extent.items()}

    # ── model (single GPU, no DDP) ────────────────────────────────────────
    oat_model_cfg = setup_oat_cfg(cfg, field_extent, logger)
    predictor = setup_oat_predictor(
        model_cfg=oat_model_cfg,
        estimated_total_step=1,
        var_names=list(cfg.dataset.obs_vars),
        transforms=[input_transform, output_transform],
        cfg=cfg,
    )
    predictor = predictor.to(_DEVICE).eval()
    mem("model on GPU")

    # ── one validation batch ──────────────────────────────────────────────
    batch = next(iter(val_loader))
    mem("after fetching batch (CPU)")

    batch = to_device(batch, _DEVICE)
    mem("batch on GPU")

    print("\n── Batch tensor sizes ──────────────────────────────────────────────────")
    def _report(obj, prefix=""):
        if isinstance(obj, torch.Tensor):
            print(f"  {prefix:<60s} {str(tuple(obj.shape)):<30s} {obj.nbytes/1e6:8.1f} MB")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _report(v, f"{prefix}.{k}" if prefix else k)
    _report(batch)
    print()

    if input_transform is not None:
        batch = input_transform(batch)
    mem("after input transform")

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
