"""Crop the COSMO-1E data to a box around Switzerland.

The crop makes a smaller copy of the data (the Swiss data pack). The NWP
archive (anemoi layout) keeps the grid points in the index box and the
variables of the extracted NWP store. It holds only the metadata and no data
chunks. The NWP values are in the extracted store. The extracted NWP store
keeps the same grid points and all its variables. The time axis and the
metadata do not change. The script writes v2 zarr stores with the source
compressor and one time step per chunk. The topography keeps the used
variables in the LV95 box of the cropped points plus a margin. The NWP
reference store keeps only its full x and y axes. PeakWeather is copied
unchanged. The source paths come only from the arguments.

Usage (from the repository root):
    python -m release.crop_swiss --src <data root> --dest <new data root>
"""

import argparse
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr
import zarr
from pyproj import Transformer

BOX = (45.59, 48.0, 5.75, 10.81)  # (south, north, west, east) in degrees
ARCHIVE = "mch-co1e-an-archive-0p01-2019-2024-1h-v1-pl13.zarr"
EXTRACTED = "cosmo-1e_rechunked/nwp_extracted.zarr"
TOPO = "cosmo-1e_rechunked/topo.zarr"
NWP_AXES = "cosmo-1e_rechunked/nwp.zarr"
PEAKWEATHER = "peakweather"
TOPO_VARS = (
    "DEM",
    "TPI_500M",
    "TPI_2000M",
    "WE_DERIVATIVE_2000M_SIGRATIO1",
    "SN_DERIVATIVE_2000M_SIGRATIO1",
)
TOPO_MARGIN_M = 5_000  # margin around the cropped NWP points, in metres
_STATS = ("mean", "stdev", "minimum", "maximum", "sums", "squares", "count")
_PER_VARIABLE = (*_STATS, "has_nans")
# Attrs that list variable names. The crop removes the names it drops.
_NAME_LISTS = ("constant_fields", "variables_with_nans")


def crop_box(
    lat: npt.NDArray[Any], lon: npt.NDArray[Any], box: tuple[float, float, float, float]
) -> tuple[slice, slice]:
    """Return the smallest (rows, cols) slices that hold all points in *box*.

    *lat* and *lon* have the shape (H, W). *box* is (south, north, west, east).
    """
    south, north, west, east = box
    inside = (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)
    rows, cols = np.flatnonzero(inside.any(axis=1)), np.flatnonzero(inside.any(0))
    if rows.size == 0:
        raise ValueError(f"no grid point in the box {box}")
    return slice(int(rows[0]), int(rows[-1]) + 1), slice(
        int(cols[0]), int(cols[-1]) + 1
    )


def point_index(rows: slice, cols: slice, width: int) -> npt.NDArray[np.int64]:
    """Return the flat indices ``row * width + col`` (row-major) as int64."""
    r = np.arange(rows.start, rows.stop, dtype=np.int64)
    c = np.arange(cols.start, cols.stop, dtype=np.int64)
    return (r[:, None] * width + c[None, :]).ravel()


def _write(group: Any, name: str, like: Any, data: npt.NDArray[Any]) -> None:
    """Write *data* to ``group[name]`` with the compressor and chunks of *like*."""
    chunks = tuple(
        max(1, min(c, s)) for c, s in zip(like.chunks, data.shape, strict=True)
    )
    group.create_dataset(
        name,
        data=data,
        chunks=chunks,
        compressor=like.compressor,
        fill_value=like.fill_value,
    )


def _crop_steps(
    src: str,
    dst: str,
    start: int,
    stop: int,
    keep: npt.NDArray[np.int64] | None,
    pts: npt.NDArray[np.int64],
) -> None:
    """Copy the time steps [start, stop) of the array *src* to *dst*.

    The function reads one time chunk at a time. Then it selects the
    variables *keep* (all when None) and the points *pts* in memory.
    """
    s, d = zarr.open_array(src, mode="r"), zarr.open_array(dst, mode="r+")
    for t in range(start, stop):
        step = s[t]
        if keep is not None:
            step = step[keep]
        d[t] = step[..., pts]


def _crop_time_parallel(
    src: Path,
    dst: Path,
    keep: npt.NDArray[np.int64] | None,
    pts: npt.NDArray[np.int64],
    workers: int,
) -> None:
    """Run _crop_steps on *workers* contiguous blocks of time steps.

    The workers start with spawn, because fork is not safe in a process
    with threads.
    """
    bounds = np.linspace(0, zarr.open_array(str(src), mode="r").shape[0], workers + 1)
    edges = bounds.astype(int).tolist()
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=spawn) as pool:
        jobs = [
            pool.submit(_crop_steps, str(src), str(dst), a, b, keep, pts)
            for a, b in pairwise(edges)
            if b > a
        ]
        for job in jobs:
            job.result()  # raise the errors of the workers


def crop_archive(
    src: Path,
    dst: Path,
    rows: slice,
    cols: slice,
    variables: list[str],
    workers: int,
    data: bool = True,
) -> None:
    """Crop the anemoi archive *src* to the index box and *variables*.

    When *data* is False, the ``data`` array has the same shape, type,
    chunks, compressor and fill value, but the function writes no chunks.
    A read of it gives the fill value. The workers do not start.

    The variables stay in the source order. The per-variable arrays and attrs
    are sliced. The ``field_shape`` attr gets the cropped shape. The
    ``_create_yaml_config`` attr is dropped. All other attrs and the
    ``_build`` group are copied unchanged.
    """
    s = zarr.open_group(str(src), mode="r")
    names = list(s.attrs["variables"])
    unknown = sorted(set(variables) - set(names))
    if unknown:
        raise KeyError(f"{src}: no variables {unknown}")
    keep = np.array([i for i, n in enumerate(names) if n in variables], np.int64)
    kept = [names[i] for i in keep]
    width = int(s.attrs["field_shape"][1])
    pts = point_index(rows, cols, width)

    d = zarr.open_group(str(dst), mode="w", zarr_version=2)
    values = s["data"]
    d.create_dataset(
        "data",
        shape=(values.shape[0], len(keep), values.shape[2], len(pts)),
        chunks=(1, len(keep), values.shape[2], len(pts)),
        dtype=values.dtype,
        compressor=values.compressor,
        fill_value=values.fill_value,
    )
    _write(d, "dates", s["dates"], s["dates"][:])
    for name in ("latitudes", "longitudes"):
        _write(d, name, s[name], s[name][:][pts])
    for name in _PER_VARIABLE:
        _write(d, name, s[name], s[name][:][keep])
    zarr.copy(s["_build"], d, name="_build")

    attrs = dict(s.attrs)
    # The build recipe holds internal paths. The pack must not show them.
    attrs.pop("_create_yaml_config", None)
    attrs["variables"] = kept
    attrs["variables_metadata"] = {n: attrs["variables_metadata"][n] for n in kept}
    for key in _NAME_LISTS:
        if key in attrs:
            attrs[key] = [n for n in attrs[key] if n in kept]
    attrs["field_shape"] = [rows.stop - rows.start, cols.stop - cols.start]
    d.attrs.update(attrs)
    if data:
        _crop_time_parallel(src / "data", dst / "data", keep, pts, workers)


def crop_extracted(
    src: Path, dst: Path, points: npt.NDArray[np.int64], workers: int
) -> None:
    """Crop the extracted NWP store *src* (T, V, N) to the flat *points*.

    All variables, the attrs and the time axis stay the same.
    """
    s = zarr.open_array(str(src), mode="r")
    d = zarr.open_array(
        str(dst),
        mode="w",
        shape=(s.shape[0], s.shape[1], len(points)),
        chunks=(1, s.shape[1], len(points)),
        dtype=s.dtype,
        compressor=s.compressor,
        fill_value=s.fill_value,
        zarr_version=2,
    )
    d.attrs.update(dict(s.attrs))
    _crop_time_parallel(src, dst, None, np.asarray(points, np.int64), workers)


def lv95_box(
    lat: npt.NDArray[Any], lon: npt.NDArray[Any]
) -> tuple[float, float, float, float]:
    """Return (xmin, xmax, ymin, ymax) in EPSG:2056 of the points *lat*, *lon*."""
    to_lv95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    x, y = to_lv95.transform(np.ravel(lon), np.ravel(lat))
    return float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))


def _axis_slice(axis: Any, low: float, high: float) -> slice:
    """Return the label slice for [low, high] in the direction of *axis*."""
    return slice(low, high) if axis[1] >= axis[0] else slice(high, low)


def _clipped(ds: xr.Dataset) -> xr.Dataset:
    """Rechunk each variable to its source chunks, clipped to its shape.

    The function also sets the zarr encoding of the chunks to the same
    values. The other encoding values (compressor, fill value) stay.
    """
    out = ds.copy()
    for name, var in ds.variables.items():
        src = var.encoding.get("chunks") or var.shape
        chunks = tuple(min(c, n) for c, n in zip(src, var.shape, strict=True))
        enc = {k: v for k, v in var.encoding.items() if k != "preferred_chunks"}
        enc["chunks"] = chunks
        if name in ds.dims:
            out[name].encoding = enc
            continue
        out[name] = var.chunk(dict(zip(var.dims, chunks, strict=True)))
        out[name].encoding = enc
    return out


def crop_topo(src: Path, dst: Path, box: tuple[float, float, float, float]) -> None:
    """Crop the topography *src* to ``TOPO_VARS`` and the LV95 *box*.

    *box* is (xmin, xmax, ymin, ymax). The coordinates of the variables
    stay. The source chunks stay where they fit in the cropped shape.
    """
    xmin, xmax, ymin, ymax = box
    ds = xr.open_zarr(src)[list(TOPO_VARS)]
    ds = ds.sel(
        x=_axis_slice(ds.x.values, xmin, xmax), y=_axis_slice(ds.y.values, ymin, ymax)
    )
    _clipped(ds).to_zarr(dst, mode="w-", zarr_format=2, consolidated=True)


def copy_nwp_axes(src: Path, dst: Path) -> None:
    """Write a zarr store with only the full ``x`` and ``y`` axes of *src*."""
    ds = xr.open_zarr(src)
    axes = xr.Dataset(coords={"x": ds.x, "y": ds.y}, attrs=ds.attrs)
    _clipped(axes).to_zarr(dst, mode="w-", zarr_format=2, consolidated=True)


def crop_all(src_root: Path, dest_root: Path, workers: int) -> None:
    """Write the 5 cropped stores below *dest_root* at the default names.

    The function checks all source and target paths before it writes. It
    raises FileNotFoundError when a source path is missing. It raises
    FileExistsError when a target path exists. The cropped archive holds only
    the metadata and no data chunks. The NWP values are in the extracted store.
    """
    targets = [ARCHIVE, EXTRACTED, TOPO, NWP_AXES, PEAKWEATHER]
    for t in targets:
        if not (src_root / t).exists():
            raise FileNotFoundError(f"{src_root / t}: source path is missing")
    found = [t for t in targets if (dest_root / t).exists()]
    if found:
        raise FileExistsError(f"{dest_root}: holds {found}")

    archive = zarr.open_group(str(src_root / ARCHIVE), mode="r")
    height, width = (int(n) for n in archive.attrs["field_shape"])
    lat = archive["latitudes"][:].reshape(height, width)
    lon = archive["longitudes"][:].reshape(height, width)
    rows, cols = crop_box(lat, lon, BOX)
    pts = point_index(rows, cols, width)
    print(
        f"rows {rows.start}:{rows.stop}, cols {cols.start}:{cols.stop},"
        f" {pts.size} points"
    )
    variables = list(
        zarr.open_array(str(src_root / EXTRACTED), mode="r").attrs["variables"]
    )
    xmin, xmax, ymin, ymax = lv95_box(lat[rows, cols], lon[rows, cols])
    m = TOPO_MARGIN_M
    topo_box = (xmin - m, xmax + m, ymin - m, ymax + m)

    (dest_root / EXTRACTED).parent.mkdir(parents=True, exist_ok=True)
    crop_archive(
        src_root / ARCHIVE, dest_root / ARCHIVE, rows, cols, variables, workers, False
    )
    crop_extracted(src_root / EXTRACTED, dest_root / EXTRACTED, pts, workers)
    crop_topo(src_root / TOPO, dest_root / TOPO, topo_box)
    copy_nwp_axes(src_root / NWP_AXES, dest_root / NWP_AXES)
    shutil.copytree(src_root / PEAKWEATHER, dest_root / PEAKWEATHER)


def main(argv: list[str] | None = None) -> None:
    """Read the command line and run crop_all."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", type=Path, required=True, help="source data root")
    parser.add_argument("--dest", type=Path, required=True, help="new data root")
    parser.add_argument("--workers", type=int, default=8, help="worker processes")
    args = parser.parse_args(argv)
    crop_all(args.src, args.dest, args.workers)


if __name__ == "__main__":
    main()
