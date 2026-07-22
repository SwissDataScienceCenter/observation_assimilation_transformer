"""Aggregate per-checkpoint test-set results into a comparison table.

Reads the checkpoint registry (``configs/checkpoints.yaml``), resolves each
entry to its ``data/results/<tag>_metrics_<split>.summary.json`` (+ per-station
``.csv``), and emits the paper's method-comparison table as LaTeX and plain
text. Rows are the registry model keys (``*_temp`` / ``*_precip`` siblings are
merged into a single method row); columns are {MAE, CRPS, Skill} per variable.

Skill is the CRPS skill score ``1 - CRPS_model / CRPS_uninformed`` relative to
the ``uninformed`` baseline of the same dataset (NA if that baseline is absent).
Uncertainties are bootstrap standard deviations over eval stations. Any value
without a backing result file renders ``NA``.

Usage
-----
python -m scripts.build_results_table [--split test] [--n-boot 1000]
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
INTERPOLATOR_SENTINEL = "interpolator"

# Canonical variables (order = table column order) and their dataset-specific
# aliases: cosmo uses temperature/precipitation, era5 uses temp/prcp.
VARIABLES = ["temperature", "precipitation"]
VAR_ALIASES = {
    "temperature": ["temperature", "temp"],
    "precipitation": ["precipitation", "prcp", "precip"],
}


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
    if key == INTERPOLATOR_SENTINEL or value == INTERPOLATOR_SENTINEL:
        return f"interpolator_{dataset}"
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


def _boot_std(
    values: "np.ndarray[Any, Any]", n_boot: int, rng: np.random.Generator
) -> float:
    """Bootstrap std of the (nan-)mean over stations."""
    v = np.asarray(values, dtype=float)
    if v.size == 0 or np.all(np.isnan(v)):
        return float("nan")
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = np.nanmean(v[idx], axis=1)
    return float(np.nanstd(means))


def _boot_skill_std(
    model_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    var: str,
    n_boot: int,
    rng: np.random.Generator,
) -> float:
    """Bootstrap std of CRPS skill, coord-paired against the reference."""
    if model_df is None or ref_df is None:
        return float("nan")
    m_col = _col(model_df, "crps", var)
    r_col = _col(ref_df, "crps", var)
    if m_col is None or r_col is None:
        return float("nan")
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
        return float("nan")
    idx = rng.integers(0, m.size, size=(n_boot, m.size))
    skill = 1.0 - np.nanmean(m[idx], axis=1) / np.nanmean(r[idx], axis=1)
    return float(np.nanstd(skill))


def _collect(
    registry: dict[str, Any], split: str, n_boot: int, seed: int
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
                    "vars": {},  # var -> {mae, crps, mae_err, crps_err, stations}
                }
                order.append(base)
            summary = _load_summary(tag, split) if tag else {}
            stations = _load_stations(tag, split) if tag else None
            for var in key_vars:
                entry = summary.get(var, {})
                mae = entry.get("mae", float("nan"))
                crps = entry.get("crps", float("nan"))
                mae_err = crps_err = float("nan")
                if stations is not None:
                    mae_col = _col(stations, "mae", var)
                    crps_col = _col(stations, "crps", var)
                    if mae_col:
                        mae_err = _boot_std(stations[mae_col].to_numpy(), n_boot, rng)
                    if crps_col:
                        crps_err = _boot_std(stations[crps_col].to_numpy(), n_boot, rng)
                rows[base]["vars"][var] = {
                    "mae": mae,
                    "crps": crps,
                    "mae_err": mae_err,
                    "crps_err": crps_err,
                    "stations": stations,
                }
        out[dataset] = [rows[b] for b in order]

    # Second pass: CRPS skill vs the dataset's uninformed baseline.
    for _dataset, method_rows in out.items():
        ref = next((r for r in method_rows if "uninformed" in r["base"]), None)
        for row in method_rows:
            for var in VARIABLES:
                v = row["vars"].get(var)
                if v is None:
                    continue
                v["skill"] = float("nan")
                v["skill_err"] = float("nan")
                if ref is None:
                    continue
                ref_v = ref["vars"].get(var, {})
                ref_crps = ref_v.get("crps", float("nan"))
                if not np.isnan(v["crps"]) and ref_crps and not np.isnan(ref_crps):
                    v["skill"] = 1.0 - v["crps"] / ref_crps
                    v["skill_err"] = _boot_skill_std(
                        v["stations"],
                        ref_v.get("stations"),
                        var,
                        n_boot,
                        np.random.default_rng(seed),
                    )
    return out


def _fmt(value: float, err: float, decimals: int, latex: bool) -> str:
    """Render 'value ± err', or NA."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    v = f"{value:.{decimals}f}"
    if err is None or (isinstance(err, float) and np.isnan(err)):
        return f"${v}$" if latex else v
    e = f"{err:.{decimals}f}"
    return f"${v} \\pm {e}$" if latex else f"{v}±{e}"


def _cells(row: dict[str, Any], latex: bool) -> list[str]:
    cells: list[str] = []
    for var in VARIABLES:
        v = row["vars"].get(var, {})
        cells.append(
            _fmt(v.get("mae", float("nan")), v.get("mae_err", float("nan")), 3, latex)
        )
        cells.append(
            _fmt(v.get("crps", float("nan")), v.get("crps_err", float("nan")), 3, latex)
        )
        cells.append(
            _fmt(
                v.get("skill", float("nan")), v.get("skill_err", float("nan")), 3, latex
            )
        )
    return cells


def _render_latex(data: dict[str, list[dict[str, Any]]]) -> str:
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
            cells = " & ".join(_cells(row, latex=True))
            lines.append(f"{lead}{row['label']} & {cells} \\\\")
        if di < len(datasets) - 1:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _render_text(data: dict[str, list[dict[str, Any]]]) -> str:
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
                [label if ri == 0 else "", row["label"], *_cells(row, latex=False)]
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
        description="Build the method-comparison results table."
    )
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--registry", default=str(_DEFAULT_REGISTRY))
    p.add_argument(
        "--results-dir", default=None, help="Override data/results location."
    )
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-prefix",
        default=str(RESULTS_DIR / "results_table"),
        help="Output path prefix; writes <prefix>.tex and <prefix>.txt.",
    )
    args = p.parse_args()

    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)

    registry = cast(
        dict[str, Any],
        OmegaConf.to_container(OmegaConf.load(args.registry), resolve=True),
    )
    data = _collect(registry, args.split, args.n_boot, args.seed)

    text = _render_text(data)
    latex = _render_latex(data)

    tex_path = Path(f"{args.out_prefix}.tex")
    txt_path = Path(f"{args.out_prefix}.txt")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(latex)
    txt_path.write_text(text)

    print(text)
    print(f"Wrote {tex_path}")
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
