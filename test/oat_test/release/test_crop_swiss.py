import numpy as np
import pytest
import zarr
from anemoi.datasets import open_dataset

from release.crop_swiss import crop_archive, crop_box, crop_extracted, point_index

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
