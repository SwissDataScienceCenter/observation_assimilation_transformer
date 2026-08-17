import torch

from oat.data_module.field import Field
from oat.data_module.transform import FieldValueTransform, StandardNormalizeTransform


def test_field_value_transform_maps_values_passes_rest():
    inner = StandardNormalizeTransform(
        mean=torch.tensor([2.0]), std=torch.tensor([2.0])
    )
    ft = FieldValueTransform(inner)
    coords = torch.tensor([[0.0, 0.0, 5.0], [1.0, 1.0, 5.0]])
    mask = torch.tensor([[1.0], [0.0]])
    field = Field(
        coords=coords, values=torch.tensor([[4.0], [6.0]]), var_names=["v"], mask=mask
    )
    out = ft(field)
    assert isinstance(out, Field)
    assert torch.allclose(out.values, torch.tensor([[1.0], [2.0]]))  # (x-2)/2
    assert out.coords is coords and out.mask is mask and out.var_names == ["v"]
    back = ft.inverse(out)
    assert torch.allclose(back.values, field.values)


def test_field_value_transform_roundtrip_dict():
    inner = StandardNormalizeTransform(
        mean=torch.tensor([0.0]), std=torch.tensor([1.0])
    )
    d = FieldValueTransform(inner).to_dict()
    from oat.data_module.transform import Transform

    rebuilt = Transform.from_dict(d)
    assert isinstance(rebuilt, FieldValueTransform)
