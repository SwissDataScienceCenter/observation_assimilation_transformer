import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from anemoi.datasets import open_dataset

from release.crop_swiss import (
    ARCHIVE,
    EXTRACTED,
    TOPO_VARS,
    copy_nwp_axes,
    crop_all,
    crop_archive,
    crop_box,
    crop_extracted,
    crop_topo,
    lv95_box,
    point_index,
)
from release.leak_check import load_pattern

GLOBAL_VARS = ["T_G", "T_2M", "U_10M", "V_10M", "TOT_PREC_1H", "CLCT"]
_STATS = ("mean", "stdev", "minimum", "maximum", "sums", "squares", "count")


def make_fake_archive(
    path,
    T=5,
    H=4,
    W=6,
    variables=("A", "T_2M", "B", "CLCT"),
    lat0=45.0,
    lon0=5.0,
    step=1.0,
):
    """Write a small anemoi archive. lat[r, c] = lat0 + r*step, lon = lon0 + c*step."""
    V, N = len(variables), H * W
    comp = zarr.Blosc(cname="lz4", clevel=5, shuffle=1)
    root = zarr.open_group(str(path), mode="w", zarr_version=2)
    t = np.arange(T)[:, None, None, None]
    v = np.arange(V)[None, :, None, None]
    p = np.arange(N)[None, None, None, :]
    data = (t * 1000 + v * 100 + p).astype("f4")
    root.create_dataset("data", data=data, chunks=(1, V, 1, N), compressor=comp)
    start = np.datetime64("2020-01-01T00:00:00", "s")
    dates = start + np.arange(T) * np.timedelta64(1, "h")
    root.create_dataset("dates", data=dates, compressor=comp)
    lat, lon = np.meshgrid(
        lat0 + step * np.arange(H), lon0 + step * np.arange(W), indexing="ij"
    )
    root.create_dataset("latitudes", data=lat.ravel(), compressor=comp)
    root.create_dataset("longitudes", data=lon.ravel(), compressor=comp)
    for k, name in enumerate(_STATS):
        vals = np.arange(V, dtype="f8") + 10.0 * (k + 1)
        dtype = "i8" if name == "count" else "f8"
        root.create_dataset(name, data=vals.astype(dtype), compressor=comp)
    root.create_dataset("has_nans", data=np.arange(V) % 2 == 0, compressor=comp)
    build = root.create_group("_build")
    build.create_dataset("flags", data=np.ones(2, bool), compressor=comp)
    build.create_dataset("lengths", data=np.array([3, T - 3], "i4"), compressor=comp)
    iso = [str(d) for d in dates]
    root.attrs.update(
        variables=list(variables),
        field_shape=[H, W],
        flatten_grid=True,
        frequency="1h",
        start_date=iso[0],
        end_date=iso[-1],
        missing_dates=[iso[2]],
        resolution=0.01,
        ensemble_dimension=1,
        variables_metadata={n: {"mars": {"param": n}} for n in variables},
        licence="unknown",
        attribution="unknown",
        version="0.30",
        constant_fields=["A"],
        # Split the string, so that the leak check of this file does not match.
        _create_yaml_config={"config_path": "/" + "users/someone/config.yaml"},
    )
    ds = open_dataset(str(path))
    assert list(ds.name_to_index) == list(variables)
    assert tuple(ds.field_shape) == (H, W) and len(ds) == T
    return path


def make_fake_extracted(path, T=5, N=24):
    """Write a small extracted NWP store (T, 6, N) with value t*1000 + v*100 + n."""
    t, v, n = np.ogrid[:T, :6, :N]
    z = zarr.open_array(
        str(path),
        mode="w",
        shape=(T, 6, N),
        chunks=(1, 6, N),
        dtype="f4",
        zarr_version=2,
    )
    z[:] = (t * 1000 + v * 100 + n).astype("f4")
    z.attrs["variables"] = GLOBAL_VARS
    return path


def make_fake_topo(path, descending=False):
    """Write a small topography store on a 10 km LV95 grid."""
    x = np.arange(2_480_000, 2_840_001, 10_000, dtype="f8")
    y = np.arange(1_070_000, 1_300_001, 10_000, dtype="f8")
    y = y[::-1] if descending else y
    shape = (len(y), len(x))
    names = (*TOPO_VARS, "SLOPE_100M_SIGRATIO1")
    rng = np.random.default_rng(0)
    data = {n: (("y", "x"), rng.random(shape, dtype="f4")) for n in names}
    yy, xx = np.meshgrid(y, x, indexing="ij")
    ds = xr.Dataset(
        data,
        coords={
            "x": x,
            "y": y,
            "latitude": (("y", "x"), 46 + (yy - 1_200_000) / 111_000),
            "longitude": (("y", "x"), 8 + (xx - 2_600_000) / 77_000),
        },
        attrs={"crs": "epsg:2056"},
    )
    ds.chunk({"y": 8, "x": 16}).to_zarr(path, zarr_format=2, consolidated=True)
    return path


@pytest.fixture(params=[False, True], ids=["y_up", "y_down"])
def fake_topo(tmp_path, request):
    return make_fake_topo(tmp_path / "topo.zarr", descending=request.param)


@pytest.fixture
def fake_root(tmp_path):
    """Write a data root. The BOX selects rows 2:7 and cols 1:11 (50 points)."""
    root = tmp_path / "root"
    (root / "cosmo-1e_rechunked").mkdir(parents=True)
    make_fake_archive(
        root / ARCHIVE, H=8, W=12, variables=("A", *GLOBAL_VARS), lon0=5.5, step=0.5
    )
    make_fake_extracted(root / EXTRACTED, N=96)
    make_fake_topo(root / "cosmo-1e_rechunked/topo.zarr")
    nwp = xr.Dataset(
        {"HSURF": (("y", "x"), np.zeros((3, 4), "f4"))},
        coords={"x": np.arange(4.0) * 1000, "y": np.arange(3.0) * 1000},
    )
    nwp.to_zarr(root / "cosmo-1e_rechunked/nwp.zarr", zarr_format=2)
    pw = root / "peakweather"
    pw.mkdir()
    pd.DataFrame({"station": ["ABO"]}).to_parquet(pw / "stations.parquet")
    (pw / "disclaimer.txt").write_text("disclaimer")
    return root


@pytest.fixture
def fake_archive(tmp_path):
    return make_fake_archive(tmp_path / "archive.zarr")


@pytest.fixture
def fake_extracted(tmp_path):
    return make_fake_extracted(tmp_path / "extracted.zarr")


def test_crop_box_gives_the_smallest_index_box():
    lat, lon = np.meshgrid(45 + np.arange(4.0), 5 + np.arange(6.0), indexing="ij")
    assert crop_box(lat, lon, (45.5, 47.0, 6.5, 8.2)) == (slice(1, 3), slice(2, 4))


def test_point_index_is_row_major():
    assert point_index(slice(1, 3), slice(2, 4), 6).tolist() == [8, 9, 14, 15]


def test_cropped_archive_opens_with_anemoi_and_keeps_metadata(fake_archive, tmp_path):
    crop_archive(
        fake_archive,
        tmp_path / "a.zarr",
        slice(1, 3),
        slice(2, 4),
        ["CLCT", "T_2M"],
        workers=2,
    )
    src, dst = open_dataset(str(fake_archive)), open_dataset(str(tmp_path / "a.zarr"))
    assert list(dst.name_to_index) == ["T_2M", "CLCT"]  # source order
    assert tuple(dst.field_shape) == (2, 2)
    assert (dst.dates == src.dates).all() and dst.missing == src.missing
    idx = point_index(slice(1, 3), slice(2, 4), 6)
    np.testing.assert_array_equal(dst.latitudes, src.latitudes[idx])
    for name in ("T_2M", "CLCT"):
        i, j = dst.name_to_index[name], src.name_to_index[name]
        np.testing.assert_array_equal(dst[3][i], src[3][j][..., idx])
        assert dst.statistics["mean"][i] == src.statistics["mean"][j]


def test_cropped_extracted_keeps_times_variables_and_order(fake_extracted, tmp_path):
    pts = np.array([8, 9, 14, 15])
    crop_extracted(fake_extracted, tmp_path / "e.zarr", pts, workers=2)
    src = zarr.open(str(fake_extracted), "r")
    dst = zarr.open(str(tmp_path / "e.zarr"), "r")
    assert dst.shape == (5, 6, 4) and dst.chunks == (1, 6, 4)
    assert dst.attrs["variables"] == src.attrs["variables"]
    np.testing.assert_array_equal(dst[:], src[:][:, :, pts])


def test_cropped_archive_has_no_internal_path_and_filters_name_lists(
    fake_archive, tmp_path
):
    dst = tmp_path / "a.zarr"
    crop_archive(fake_archive, dst, slice(1, 3), slice(2, 4), ["T_2M"], workers=1)
    assert load_pattern().search((fake_archive / ".zattrs").read_bytes())
    assert not load_pattern().search((dst / ".zattrs").read_bytes())
    attrs = zarr.open_group(str(dst), "r").attrs
    assert "_create_yaml_config" not in attrs
    assert attrs["constant_fields"] == []


def test_lv95_box_of_bern():
    xmin, xmax, ymin, ymax = lv95_box(np.array([46.95, 46.95]), np.array([7.44, 7.44]))
    assert abs(xmin - 2_600_000) < 2_000 and abs(ymin - 1_200_000) < 2_000
    assert xmin == xmax and ymin == ymax


def test_crop_topo_keeps_only_the_used_variables_in_the_box(fake_topo, tmp_path):
    crop_topo(
        fake_topo, tmp_path / "t.zarr", (2_500_000, 2_600_000, 1_100_000, 1_200_000)
    )
    ds = xr.open_zarr(tmp_path / "t.zarr")
    assert set(ds.data_vars) == set(TOPO_VARS)
    assert {"latitude", "longitude"} <= set(ds.coords)
    assert ds.x.min() >= 2_500_000 and ds.x.max() <= 2_600_000
    assert ds.y.min() >= 1_100_000 and ds.y.max() <= 1_200_000
    assert ds.sizes == {"y": 11, "x": 11}
    src = xr.open_zarr(fake_topo).sel(x=ds.x, y=ds.y)
    np.testing.assert_array_equal(ds.DEM, src.DEM)


def test_copy_nwp_axes_keeps_full_x_and_y_only(fake_root, tmp_path):
    copy_nwp_axes(fake_root / "cosmo-1e_rechunked/nwp.zarr", tmp_path / "n.zarr")
    src = xr.open_zarr(fake_root / "cosmo-1e_rechunked/nwp.zarr")
    dst = xr.open_zarr(tmp_path / "n.zarr")
    assert not dst.data_vars
    np.testing.assert_array_equal(dst.x, src.x)
    np.testing.assert_array_equal(dst.y, src.y)


def test_crop_all_writes_the_default_names_and_a_consistent_grid(fake_root, tmp_path):
    out = tmp_path / "out"
    crop_all(fake_root, out, workers=1)
    a = open_dataset(str(out / ARCHIVE))
    e = zarr.open(str(out / EXTRACTED), "r")
    assert (
        e.shape[2]
        == len(a.latitudes)
        < len(open_dataset(str(fake_root / ARCHIVE)).latitudes)
    )
    assert e.shape[2] == 50 and list(a.name_to_index) == GLOBAL_VARS
    assert (out / "peakweather/disclaimer.txt").is_file()
    assert set(xr.open_zarr(out / "cosmo-1e_rechunked/topo.zarr").data_vars) == set(
        TOPO_VARS
    )
    assert not xr.open_zarr(out / "cosmo-1e_rechunked/nwp.zarr").data_vars


def test_crop_all_refuses_an_existing_target(fake_root, tmp_path):
    (tmp_path / "out/peakweather").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        crop_all(fake_root, tmp_path / "out", workers=1)
    assert not (tmp_path / "out" / ARCHIVE).exists()


def test_crop_all_refuses_a_missing_source(fake_root, tmp_path):
    shutil.rmtree(fake_root / "cosmo-1e_rechunked/topo.zarr")
    with pytest.raises(FileNotFoundError, match=r"topo\.zarr"):
        crop_all(fake_root, tmp_path / "out", workers=1)
    assert not (tmp_path / "out").exists()
