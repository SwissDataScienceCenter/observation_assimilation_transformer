"""Crop the COSMO-1E data to a box around Switzerland.

The crop makes a smaller copy of the data (the Swiss data pack). The NWP
archive (anemoi layout) keeps the grid points in the index box and the
selected variables. The extracted NWP store keeps the same grid points and
all its variables. The time axis and the metadata do not change. The script
writes v2 zarr stores with the source compressor and one time step per
chunk. The source paths come only from the arguments.

Usage (from the repository root):
    python -m release.crop_swiss --src <data root> --dest <new data root>
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import zarr

BOX = (45.59, 48.0, 5.75, 10.81)  # (south, north, west, east) in degrees
ARCHIVE = "mch-co1e-an-archive-0p01-2019-2024-1h-v1-pl13.zarr"
EXTRACTED = "cosmo-1e_rechunked/nwp_extracted.zarr"
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
) -> None:
    """Crop the anemoi archive *src* to the index box and *variables*.

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
    data = s["data"]
    d.create_dataset(
        "data",
        shape=(data.shape[0], len(keep), data.shape[2], len(pts)),
        chunks=(1, len(keep), data.shape[2], len(pts)),
        dtype=data.dtype,
        compressor=data.compressor,
        fill_value=data.fill_value,
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
