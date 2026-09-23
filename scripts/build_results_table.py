"""Aggregate per-checkpoint test-set results into comparison tables.

Reads the checkpoint registry (``configs/checkpoints.yaml``), resolves each
entry to its ``data/results/<tag>_metrics_<split>.summary.json`` (+ per-station
``.csv``), and emits two method-comparison tables (each as LaTeX and plain
text): a **means** table (point estimates, for the main text) and a **CI**
table (point estimate + bootstrap confidence interval, for the appendix). Rows
are the registry model keys (``*_temp`` / ``*_precip`` siblings are merged into
a single method row); columns are {MAE, CRPS, Skill} per variable.

Skill is the CRPS skill score ``1 - CRPS_model / CRPS_uninformed`` relative to
the ``uninformed`` baseline of the same dataset (NA if that baseline is absent),
computed on the stations shared by model and reference.

Bootstrap
---------
Confidence intervals are percentile intervals from a bootstrap that resamples
**eval stations** with replacement (each station's value is its metric averaged
over all timesteps). Skill is bootstrapped **paired**: the same resampled
stations are used for the model and the reference, and the skill point estimate
is computed on that same shared-station set so the interval brackets it.

Caveats (by construction):
- This estimates *spatial-generalisation* uncertainty; each station's
  time-mean is treated as fixed (temporal sampling variability is not captured).
- Stations are treated as independent draws; spatial correlation is ignored, so
  the intervals are likely optimistic (too narrow).
- MAE/CRPS intervals are *marginal* per method; because all methods share the
  same stations/times their errors are correlated, so overlapping MAE/CRPS
  intervals do NOT imply the paired difference is insignificant. The Skill
  column (paired vs uninformed) is the significance-aware comparison.

Any value without a backing result file renders ``NA``.

Usage
-----
python -m scripts.build_results_table [--split test] [--n-boot 1000] [--ci-level 95]
"""

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# Same registry path as oat.predictor_module.checkpoint_utils._DEFAULT_REGISTRY,
# resolved directly to avoid importing the heavy oat/torch stack for one Path.
_DEFAULT_REGISTRY = Path(__file__).parents[1] / "configs" / "checkpoints.yaml"
RESULTS_DIR = Path(__file__).parents[1] / "data/results"

# Canonical variables (order = table column order) and their dataset-specific
# aliases: cosmo uses temperature/precipitation, era5 uses temp/prcp.
VARIABLES = ["temperature", "precipitation"]
VAR_ALIASES = {
    "temperature": ["temperature", "temp"],
    "precipitation": ["precipitation", "prcp", "precip"],
}

# Metrics shown per variable and their point/CI keys in a var dict.
METRICS = ["mae", "crps", "skill"]

NAN = float("nan")
CI = tuple[float, float]


def _col(df: pd.DataFrame, metric: str, var: str) -> str | None:
    """Column ``{metric}_{alias}`` present in ``df`` for canonical ``var``, or None."""
    for alias in VAR_ALIASES[var]:
        col = f"{metric}_{alias}"
        if col in df.columns:
            return col
    return None


# Pretty labels; fall back to the raw base key when absent.
MODEL_LABELS = {
    "interpolator": "NWP Interp.",
    "cosmo_interpolator": "COSMO NWP Interp.",
    "uninformed": "Uninformed baseline",
    "gp": "Gaussian Process",
    "deep_gp": "Deep Gaussian Process",
    "ootg": "Gridded TNP",
    "oat": "OAT (observation assim.)",
    "oat_vanilla": "OAT (vanilla attn.)",
    "oat_tetnp": "OAT (TETNP attn.)",
}
DATASET_LABELS = {
    "cosmo_peakweather": "Switzerland",
    "era5_meteostat": "Europe",
}


def _base_and_vars(key: str) -> tuple[str, list[str]]:
    """Map a registry key to (row base name, variables it carries)."""
    if key.endswith("_temp_precip"):
        return key[: -len("_temp_precip")], ["temperature", "precipitation"]
    if key.endswith("_temp"):
        return key[: -len("_temp")], ["temperature"]
    if key.endswith("_precip"):
        return key[: -len("_precip")], ["precipitation"]
    return key, ["temperature", "precipitation"]


def _resolve_tag(dataset: str, key: str, value: Any) -> str | None:
    """Result-file tag for a registry entry (dir name, mirroring evaluate_test_set)."""
    if key == value:  # sentinel: the model key stands in for a checkpoint path
        return f"{key}_{dataset}"
    if value is None:
        return None
    return Path(str(value)).parent.name


def _load_summary(tag: str, split: str) -> dict[str, dict[str, float]]:
    """Return {var: {'mae': float, 'crps': float}} from the summary JSON, or {}."""
    path = RESULTS_DIR / f"{tag}_metrics_{split}.summary.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out: dict[str, dict[str, float]] = {}
    for var in VARIABLES:
        entry = {}
        for metric in ("mae", "crps"):
            block = data.get(metric, {})
            for alias in VAR_ALIASES[var]:
                if alias in block:
                    entry[metric] = float(block[alias])
                    break
        if entry:
            out[var] = entry
    return out


def _load_stations(tag: str, split: str) -> pd.DataFrame | None:
    """Per-station CSV (station_x, station_y, crps_<var>, mae_<var>), or None."""
    path = RESULTS_DIR / f"{tag}_metrics_{split}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _boot_ci(
    values: "np.ndarray[Any, Any]",
    n_boot: int,
    rng: np.random.Generator,
    pct: tuple[float, float],
) -> CI:
    """Percentile CI of the mean over stations (resampling stations)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (NAN, NAN)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, pct)
    return (float(lo), float(hi))


def _paired_crps(
    model_df: pd.DataFrame | None,
    ref_df: pd.DataFrame | None,
    var: str,
) -> tuple["np.ndarray[Any, Any]", "np.ndarray[Any, Any]"] | None:
    """Coord-paired, NaN-dropped per-station CRPS arrays (model, reference)."""
    if model_df is None or ref_df is None:
        return None
    m_col = _col(model_df, "crps", var)
    r_col = _col(ref_df, "crps", var)
    if m_col is None or r_col is None:
        return None
    merged = model_df[["station_x", "station_y", m_col]].merge(
        ref_df[["station_x", "station_y", r_col]],
        on=["station_x", "station_y"],
        suffixes=("_m", "_r"),
    )
    cols = list(merged.columns[2:])  # the two crps columns after the coords
    m = merged[cols[0]].to_numpy(dtype=float)
    r = merged[cols[1]].to_numpy(dtype=float)
    mask = ~np.isnan(m) & ~np.isnan(r)
    m, r = m[mask], r[mask]
    if m.size == 0:
        return None
    return m, r


def _boot_skill(
    model_df: pd.DataFrame | None,
    ref_df: pd.DataFrame | None,
    var: str,
    n_boot: int,
    rng: np.random.Generator,
    pct: tuple[float, float],
) -> tuple[float, CI]:
    """Paired CRPS skill point estimate and percentile CI (shared stations)."""
    paired = _paired_crps(model_df, ref_df, var)
    if paired is None:
        return NAN, (NAN, NAN)
    m, r = paired
    r_point = r.mean()
    if not np.isfinite(r_point) or r_point <= 0:
        return NAN, (NAN, NAN)
    point = 1.0 - m.mean() / r_point
    idx = rng.integers(0, m.size, size=(n_boot, m.size))
    r_boot = r[idx].mean(axis=1)
    r_boot = np.maximum(r_boot, 1e-12)
    skill = 1.0 - m[idx].mean(axis=1) / r_boot
    lo, hi = np.percentile(skill, pct)
    return float(point), (float(lo), float(hi))


def _collect(
    registry: dict[str, Any],
    split: str,
    n_boot: int,
    seed: int,
    pct: tuple[float, float],
) -> dict[str, list[dict[str, Any]]]:
    """Build {dataset: [row, ...]} of aggregated per-method metrics."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[dict[str, Any]]] = {}
    for dataset, models in registry.items():
        if models is None:
            continue
        rows: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for key, value in models.items():
            base, key_vars = _base_and_vars(key)
            tag = _resolve_tag(dataset, key, value)
            if base not in rows:
                rows[base] = {
                    "base": base,
                    "label": MODEL_LABELS.get(base, base),
                    "vars": {},
                }
                order.append(base)
            summary = _load_summary(tag, split) if tag else {}
            stations = _load_stations(tag, split) if tag else None
            for var in key_vars:
                entry = summary.get(var, {})
                # Per-variable stations: gp/deep_gp merge separate _temp/_precip
                # CSVs, so each variable keeps the file that actually holds it.
                v: dict[str, Any] = {
                    "mae": entry.get("mae", NAN),
                    "crps": entry.get("crps", NAN),
                    "mae_ci": (NAN, NAN),
                    "crps_ci": (NAN, NAN),
                    "skill": NAN,
                    "skill_ci": (NAN, NAN),
                    "stations": stations,
                }
                if stations is not None:
                    mae_col = _col(stations, "mae", var)
                    crps_col = _col(stations, "crps", var)
                    if mae_col:
                        v["mae_ci"] = _boot_ci(
                            stations[mae_col].to_numpy(), n_boot, rng, pct
                        )
                    if crps_col:
                        v["crps_ci"] = _boot_ci(
                            stations[crps_col].to_numpy(), n_boot, rng, pct
                        )
                rows[base]["vars"][var] = v
        out[dataset] = [rows[b] for b in order]

    # Second pass: paired CRPS skill vs the dataset's uninformed baseline.
    for method_rows in out.values():
        ref = next((r for r in method_rows if "uninformed" in r["base"]), None)
        for row in method_rows:
            for var in VARIABLES:
                v = row["vars"].get(var)
                if v is None or ref is None:
                    continue
                ref_v = ref["vars"].get(var, {})
                v["skill"], v["skill_ci"] = _boot_skill(
                    v["stations"], ref_v.get("stations"), var, n_boot, rng, pct
                )
    return out


def _fmt_mean(value: float, decimals: int, latex: bool) -> str:
    if value is None or np.isnan(value):
        return "NA"
    s = f"{value:.{decimals}f}"
    return f"${s}$" if latex else s


def _fmt_ci(value: float, ci: CI, decimals: int, latex: bool) -> str:
    """Interval only -- point estimates live in the means table."""
    lo, hi = ci
    if value is None or np.isnan(value) or np.isnan(lo) or np.isnan(hi):
        return "NA"
    return f"[{lo:.{decimals}f}, {hi:.{decimals}f}]"


def _cells(row: dict[str, Any], mode: str, latex: bool) -> list[str]:
    cells: list[str] = []
    for var in VARIABLES:
        v = row["vars"].get(var, {})
        for metric in METRICS:
            value = v.get(metric, NAN)
            if mode == "means":
                cells.append(_fmt_mean(value, 3, latex))
            else:
                cells.append(
                    _fmt_ci(value, v.get(f"{metric}_ci", (NAN, NAN)), 3, latex)
                )
    return cells


def _render_latex(data: dict[str, list[dict[str, Any]]], mode: str) -> str:
    lines = [
        r"\begin{tabular}{ll|ccc|ccc}",
        r"    \toprule",
        r"    & & \multicolumn{3}{c|}{\textbf{Temperature}} & "
        r"\multicolumn{3}{c}{\textbf{Precipitation}} \\",
        r"    \cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"    \textbf{Dataset} & \textbf{Method} "
        r"& MAE $\downarrow$ & CRPS $\downarrow$ & Skill $\uparrow$ "
        r"& MAE $\downarrow$ & CRPS $\downarrow$ & Skill $\uparrow$\\",
        r"    \midrule",
    ]
    datasets = list(data.keys())
    for di, dataset in enumerate(datasets):
        method_rows = data[dataset]
        label = DATASET_LABELS.get(dataset, dataset)
        n = len(method_rows)
        for ri, row in enumerate(method_rows):
            lead = (
                rf"    \multirow{{{n}}}{{*}}{{\textbf{{{label}}}}} & "
                if ri == 0
                else r"    & "
            )
            cells = " & ".join(_cells(row, mode, latex=True))
            lines.append(f"{lead}{row['label']} & {cells} \\\\")
        if di < len(datasets) - 1:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _render_text(data: dict[str, list[dict[str, Any]]], mode: str) -> str:
    header = [
        "Dataset",
        "Method",
        "T-MAE",
        "T-CRPS",
        "T-Skill",
        "P-MAE",
        "P-CRPS",
        "P-Skill",
    ]
    table_rows: list[list[str]] = []
    for dataset, method_rows in data.items():
        label = DATASET_LABELS.get(dataset, dataset)
        for ri, row in enumerate(method_rows):
            table_rows.append(
                [
                    label if ri == 0 else "",
                    row["label"],
                    *_cells(row, mode, latex=False),
                ]
            )
    widths = [
        max(len(header[c]), *(len(r[c]) for r in table_rows))
        for c in range(len(header))
    ]

    def _fmt_row(cells: list[str]) -> str:
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    out = [_fmt_row(header), "-+-".join("-" * w for w in widths)]
    prev = None
    for r in table_rows:
        if prev is not None and r[0] and r[0] != prev:
            out.append("-+-".join("-" * w for w in widths))
        out.append(_fmt_row(r))
        if r[0]:
            prev = r[0]
    return "\n".join(out) + "\n"


def main() -> None:
    global RESULTS_DIR
    p = argparse.ArgumentParser(
        description="Build the method-comparison results tables (means + CI)."
    )
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--registry", default=str(_DEFAULT_REGISTRY))
    p.add_argument(
        "--results-dir", default=None, help="Override data/results location."
    )
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ci-level", type=float, default=95.0, help="Two-sided CI level in percent."
    )
    p.add_argument(
        "--out-prefix",
        default=None,
        help="Output prefix; writes <prefix>_means.* and <prefix>_ci.* (.tex, .txt).",
    )
    args = p.parse_args()

    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)

    if args.out_prefix is None:
        args.out_prefix = str(RESULTS_DIR / "results_table")
    tail = (100.0 - args.ci_level) / 2.0
    pct = (tail, 100.0 - tail)

    registry = cast(
        dict[str, Any],
        OmegaConf.to_container(OmegaConf.load(args.registry), resolve=True),
    )
    data = _collect(registry, args.split, args.n_boot, args.seed, pct)

    outputs = {
        "means": "point estimates (main text)",
        "ci": f"{args.ci_level:g}% CI (appendix)",
    }
    for mode, desc in outputs.items():
        tex_path = Path(f"{args.out_prefix}_{mode}.tex")
        txt_path = Path(f"{args.out_prefix}_{mode}.txt")
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.write_text(_render_latex(data, mode))
        txt_path.write_text(_render_text(data, mode))
        print(f"=== {desc} ===")
        print(_render_text(data, mode))
        print(f"Wrote {tex_path}")
        print(f"Wrote {txt_path}\n")


if __name__ == "__main__":
    main()
