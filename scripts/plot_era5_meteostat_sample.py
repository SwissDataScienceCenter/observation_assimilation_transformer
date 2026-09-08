"""Plot one data point from ERA5MeteostatDataset.

Usage:
    python -m scripts.plot_era5_meteostat_sample
    python -m scripts.plot_era5_meteostat_sample --start_date 2022-07-15 --end_date 2022-07-15
    python -m scripts.plot_era5_meteostat_sample --sample_idx 10 --out my_plot.png
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from omegaconf import OmegaConf

from oat.data_module.legacy.era5_meteostat_dataset import ERA5MeteostatDataset

_CFG_DIR = Path(__file__).parent.parent / "configs" / "dataset" / "era5_meteostat"


def _load_cfg(start_date=None, end_date=None) -> OmegaConf:
    settings = OmegaConf.load(_CFG_DIR / "settings.yaml")
    catalog = OmegaConf.load(_CFG_DIR / "catalog.yaml")
    defaults = OmegaConf.load(_CFG_DIR / "default.yaml")
    # drop the Hydra defaults list before merging
    defaults_clean = {k: v for k, v in defaults.items() if k != "defaults"}
    cfg = OmegaConf.merge(settings, OmegaConf.create(defaults_clean))
    cfg.catalog = catalog
    if start_date is not None:
        cfg.start_date = start_date
    if end_date is not None:
        cfg.end_date = end_date

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--out", default="era5_meteostat_sample.png")
    parser.add_argument("--start_date", default=None)
    parser.add_argument("--end_date", default=None)
    args = parser.parse_args()

    cfg = _load_cfg(start_date=args.start_date, end_date=args.end_date)
    sample_idx = args.sample_idx
    out_path = args.out

    print(f"Loading dataset (start={cfg.get('start_date')}, end={cfg.get('end_date')}) ...")
    ds = ERA5MeteostatDataset(cfg)
    print(f"  {len(ds)} timesteps, {len(ds.stations)} stations")

    sample = ds[sample_idx]
    time = ds.times[sample_idx]

    global_field = sample["global_data"]["global_field"]  # [C, H, W]
    coords = sample["observation_data"]["coords"]          # [N, 2]  (lon, lat)
    obs = sample["observation_data"]["obs"]                # [N, V]
    mask = sample["observation_data"]["mask_obs"]          # [N, V]  1=valid

    nwp_vars = list(cfg.nwp_vars)
    obs_vars = list(cfg.obs_vars)

    lons = ds._ref_lons
    lats = ds._ref_lats

    n_panels = 1 + len(obs_vars)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    # --- Panel 0: first NWP variable (2t → convert K→°C if present) ---
    nwp_idx = 0
    field = global_field[nwp_idx].numpy()
    label = nwp_vars[nwp_idx]
    unit = "K"
    if label == "2t":
        field = field - 273.15
        label = "2m temperature"
        unit = "°C"

    im = axes[0].pcolormesh(lons, lats, field, cmap="RdBu_r", shading="auto")
    fig.colorbar(im, ax=axes[0], label=unit, shrink=0.8)
    axes[0].set_title(f"{label}\n{time.strftime('%Y-%m-%d %H:%M UTC')}")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    # --- Panels 1+: each obs variable as scatter over the NWP background ---
    for v_idx, var in enumerate(obs_vars):
        ax = axes[1 + v_idx]

        valid = mask[:, v_idx].bool().numpy()
        lon_v = coords[valid, 0].numpy()
        lat_v = coords[valid, 1].numpy()
        val_v = obs[valid, v_idx].numpy()

        # faint NWP background
        ax.pcolormesh(lons, lats, field, cmap="Greys", shading="auto", alpha=0.3)

        sc = ax.scatter(lon_v, lat_v, c=val_v, s=6, cmap="plasma",
                        norm=mcolors.CenteredNorm(halfrange=np.nanstd(val_v) * 2.5)
                        if var not in ("prcp", "wspd", "wpgt", "rhum")
                        else None)
        fig.colorbar(sc, ax=ax, shrink=0.8, label=_unit(var))
        ax.set_title(f"Stations: {var}  ({valid.sum()}/{len(valid)} valid)")
        ax.set_xlabel("Longitude")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


def _unit(var: str) -> str:
    return {"temp": "°C", "rhum": "%", "prcp": "mm/h", "wspd": "m/s", "wpgt": "m/s"}.get(var, "")


if __name__ == "__main__":
    main()