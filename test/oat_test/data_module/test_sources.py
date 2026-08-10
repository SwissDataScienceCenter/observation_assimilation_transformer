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
