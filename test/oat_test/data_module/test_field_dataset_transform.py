import torch

from oat.data_module.transform import FieldValueTransform, NestedTransform
from oat_test.data_module.test_field_dataset_base import _build_dataset


def test_get_transform_real_standardizes_all_and_logs_obs():
    ds, _ = _build_dataset()  # transform_type == "real"
    post, output = ds.get_transform(num_data_points=2)
    assert isinstance(post, NestedTransform)
    fields_tr = post.transform_dict["fields"].transform_dict
    # every field has a FieldValueTransform leaf
    assert set(fields_tr.keys()) == {"obs", "nwp", "geo"}
    for leaf in fields_tr.values():
        assert isinstance(leaf, FieldValueTransform)
    # applying to a batch standardizes values and preserves coords/mask
    sample = ds[0]
    out = post({"fields": sample["fields"]})
    for name, f in out["fields"].items():
        assert f.coords is sample["fields"][name].coords
    # output_transform round-trips obs values
    obs_vals = sample["fields"]["obs"].values
    normed = output(obs_vals)
    assert torch.allclose(output.inverse(normed), obs_vals, atol=1e-4)


def test_baseline_pre_transform_maps_nwp_to_obs_units():
    ds, _ = _build_dataset()
    # temperature obs (degC) from nwp T_2M (Kelvin): subtract 273.15
    mapping = {"temperature": (["T_2M"], lambda nwp: nwp["T_2M"] - 273.15)}
    pre = ds.get_baseline_pre_transform(nwp_field="nwp", obs_to_nwp=mapping)
    sample = ds[0]
    new = pre({"fields": sample["fields"]})
    expected = sample["fields"]["nwp"].values[:, 0] - 273.15
    assert torch.allclose(new["fields"]["nwp"].values[:, 0], expected)
