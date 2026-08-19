import torch

from oat.data_module.sources import PointSetSource, RasterSource, SourceInfo


def test_pointset_nearest_and_identity():
    coords = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    vals = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # [C=1, N=4]
    src = PointSetSource(coords, ["v"], lambda t: vals, native_coord_source="pts")
    # identity: querying the source's own coords returns arange
    idx = src.build_index(coords)
    assert torch.equal(idx, torch.arange(4))
    # nearest: a point near corner 3 (index 3) selects it
    idx2 = src.build_index(torch.tensor([[9.0, 9.0]]))
    assert idx2.tolist() == [3]
    frame = src.load(0)  # [C, N]
    assert frame[0][idx2].item() == 4.0
    assert not src.has_mask and src.load_mask(0) is None


def test_pointset_mask_and_info():
    coords = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    src = PointSetSource(
        coords,
        ["temperature"],
        values_fn=lambda t: torch.tensor([[5.0, 6.0]]),
        mask_fn=lambda t: torch.tensor([[1.0, 0.0]]),
        units={"temperature": "degC"},
        native_coord_source="stations",
    )
    assert src.has_mask
    idx = src.build_index(coords)
    assert src.load_mask(0)[0][idx].tolist() == [1.0, 0.0]
    assert src.info == SourceInfo(
        available_vars=["temperature"],
        native_coords="stations",
        has_mask=True,
        units={"temperature": "degC"},
    )


def test_raster_index_arithmetic():
    # 1 channel, 3x3 grid, origin (0,0), res (1,1); raster[c, row=y, col=x]
    raster = torch.arange(9.0).reshape(1, 3, 3)  # value = y*3 + x
    src = RasterSource(raster, origin=(0.0, 0.0), res=(1.0, 1.0), var_names=["dem"])
    # point (x=2, y=1) -> flat index 1*3 + 2 = 5 -> value 5
    idx = src.build_index(torch.tensor([[2.0, 1.0]]))
    assert src.load(0)[0][idx].item() == 5.0
    # out-of-range point clamps into the grid (no crash)
    idx2 = src.build_index(torch.tensor([[99.0, -99.0]]))
    assert src.load(0)[0][idx2].item() == 2.0  # clamps to (x=2, y=0) -> 2


def test_pointset_build_interp_barycentric_weights():
    coords = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    vals = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # [C=1, N=4]
    src = PointSetSource(coords, ["v"], lambda t: vals, native_coord_source="pts")

    idx, w = src.build_interp(coords)
    assert idx.shape == (4, 3)
    assert w.shape == (4, 3)
    assert torch.allclose(w.sum(dim=-1), torch.ones(4))

    # query exactly on a source point -> that point gets weight ~1
    idx3, w3 = src.build_interp(torch.tensor([[10.0, 10.0]]))
    gathered = vals[0][idx3[0]]
    interpolated = (gathered * w3[0]).sum()
    assert abs(interpolated.item() - 4.0) < 1e-6


def test_pointset_build_interp_midpoint():
    coords = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    vals = torch.tensor([[0.0, 10.0, 20.0]])  # [C=1, N=3]
    src = PointSetSource(coords, ["v"], lambda t: vals)

    # Centroid of the triangle: barycentric weights = (1/3, 1/3, 1/3)
    centroid = torch.tensor([[10.0 / 3, 10.0 / 3]])
    idx, w = src.build_interp(centroid)
    interpolated = (vals[0][idx[0]] * w[0]).sum()
    assert abs(interpolated.item() - 10.0) < 1e-4


def test_pointset_build_interp_outside_hull_fallback():
    coords = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    vals = torch.tensor([[0.0, 10.0, 20.0]])
    src = PointSetSource(coords, ["v"], lambda t: vals)

    # Point far outside the hull — should fall back to nearest
    outside = torch.tensor([[-100.0, -100.0]])
    idx, w = src.build_interp(outside)
    interpolated = (vals[0][idx[0]] * w[0]).sum()
    assert interpolated.item() == 0.0  # nearest to (0,0) -> value 0


def test_raster_build_interp_bilinear_midpoint():
    raster = torch.arange(9.0).reshape(1, 3, 3)  # value = y*3 + x
    src = RasterSource(raster, origin=(0.0, 0.0), res=(1.0, 1.0), var_names=["dem"])

    # (0.5, 0.0) is midway between col=0 and col=1 at row=0
    # bilinear: w00=0.5, w10=0.5, w01=0, w11=0 -> mean of values 0 and 1
    idx, w = src.build_interp(torch.tensor([[0.5, 0.0]]))
    assert idx.shape == (1, 4)
    assert w.shape == (1, 4)
    assert torch.allclose(w.sum(dim=-1), torch.ones(1))
    flat = src.load(0)[0]
    interpolated = (flat[idx] * w).sum(dim=-1)
    assert abs(interpolated.item() - 0.5) < 1e-6


def test_raster_build_interp_bilinear_center():
    raster = torch.arange(9.0).reshape(1, 3, 3)
    src = RasterSource(raster, origin=(0.0, 0.0), res=(1.0, 1.0), var_names=["dem"])

    # (0.5, 0.5) is center of the bottom-left 2x2 cell
    # corners: (0,0)=0, (1,0)=1, (0,1)=3, (1,1)=4 -> mean = 2.0
    idx, w = src.build_interp(torch.tensor([[0.5, 0.5]]))
    flat = src.load(0)[0]
    interpolated = (flat[idx] * w).sum(dim=-1)
    assert abs(interpolated.item() - 2.0) < 1e-6
    assert torch.allclose(w[0], torch.tensor([0.25, 0.25, 0.25, 0.25]))


def test_raster_build_interp_on_grid_point():
    raster = torch.arange(9.0).reshape(1, 3, 3)
    src = RasterSource(raster, origin=(0.0, 0.0), res=(1.0, 1.0), var_names=["dem"])

    # Exactly on grid point (1, 1) -> value 4.0
    idx, w = src.build_interp(torch.tensor([[1.0, 1.0]]))
    flat = src.load(0)[0]
    interpolated = (flat[idx] * w).sum(dim=-1)
    assert abs(interpolated.item() - 4.0) < 1e-6
