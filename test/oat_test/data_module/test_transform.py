import pytest
import torch

from oat.data_module.support import VariableSupport
from oat.data_module.transform import (
    Transform,
    IdentityTransform,
    TranslateScaleTransform,
    StandardNormalizeTransform,
    MinMaxTransform,
    LogTransform,
    ComposedTransform,
    NestedTransform,
    transform_dict_to_yaml,
    transform_dict_from_yaml,
    TRANSFORM_REGISTRY,
)


class TestIdentityTransform:
    def test_identity_forward(self) -> None:
        transform = IdentityTransform()
        sample = torch.tensor([1.0, 2.0, 3.0])
        result = transform(sample)
        assert torch.allclose(result, sample)

    def test_identity_inverse(self) -> None:
        transform = IdentityTransform()
        sample = torch.tensor([1.0, 2.0, 3.0])
        result = transform.inverse(sample)
        assert torch.allclose(result, sample)

    def test_identity_serialization(self) -> None:
        transform = IdentityTransform()
        d = transform.to_dict()
        assert d == {"type": "IdentityTransform"}
        restored = IdentityTransform.from_dict(d)
        assert isinstance(restored, IdentityTransform)


class TestTranslateScaleTransform:
    def test_translate_scale_forward(self) -> None:
        translate = torch.tensor([1.0, 2.0])
        scale = torch.tensor([0.5, 2.0])
        transform = TranslateScaleTransform(translate, scale)
        sample = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = transform(sample)
        expected = torch.tensor([[1.0, 8.0], [2.0, 12.0]])
        assert torch.allclose(result, expected)

    def test_translate_scale_inverse(self) -> None:
        translate = torch.tensor([1.0, 2.0])
        scale = torch.tensor([0.5, 2.0])
        transform = TranslateScaleTransform(translate, scale)
        sample = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = transform.inverse(transform(sample))
        assert torch.allclose(result, sample, atol=1e-6)

    def test_translate_scale_serialization(self) -> None:
        translate = torch.tensor([1.0, 2.0])
        scale = torch.tensor([0.5, 2.0])
        transform = TranslateScaleTransform(translate, scale, channel_dim=1)
        d = transform.to_dict()
        restored = TranslateScaleTransform.from_dict(d)
        assert torch.allclose(restored.translate, translate)
        assert torch.allclose(restored.scale, scale)
        assert restored.channel_dim == 1


class TestStandardNormalizeTransform:
    def test_normalize_forward(self) -> None:
        mean = torch.tensor([1.0, 2.0])
        std = torch.tensor([0.5, 1.0])
        transform = StandardNormalizeTransform(mean, std)
        sample = torch.tensor([[1.5, 3.0], [2.0, 4.0]])
        result = transform(sample)
        expected = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        assert torch.allclose(result, expected)

    def test_normalize_inverse(self) -> None:
        mean = torch.tensor([1.0, 2.0])
        std = torch.tensor([0.5, 1.0])
        transform = StandardNormalizeTransform(mean, std)
        sample = torch.tensor([[1.5, 3.0], [2.0, 4.0]])
        result = transform.inverse(transform(sample))
        assert torch.allclose(result, sample, atol=1e-6)

    def test_normalize_serialization(self) -> None:
        mean = torch.tensor([1.0, 2.0])
        std = torch.tensor([0.5, 1.0])
        transform = StandardNormalizeTransform(mean, std)
        d = transform.to_dict()
        restored = StandardNormalizeTransform.from_dict(d)
        assert torch.allclose(restored.mean, mean)
        assert torch.allclose(restored.std, std)


class TestMinMaxTransform:
    def test_minmax_forward(self) -> None:
        data_min = torch.tensor([0.0, 10.0])
        data_max = torch.tensor([10.0, 20.0])
        transform = MinMaxTransform(data_min, data_max)
        sample = torch.tensor([[5.0, 15.0], [10.0, 20.0]])
        result = transform(sample)
        expected = torch.tensor([[0.5, 0.5], [1.0, 1.0]])
        assert torch.allclose(result, expected)

    def test_minmax_inverse(self) -> None:
        data_min = torch.tensor([0.0, 10.0])
        data_max = torch.tensor([10.0, 20.0])
        transform = MinMaxTransform(data_min, data_max)
        sample = torch.tensor([[5.0, 15.0], [10.0, 20.0]])
        result = transform.inverse(transform(sample))
        assert torch.allclose(result, sample, atol=1e-6)

    def test_minmax_serialization(self) -> None:
        data_min = torch.tensor([0.0, 10.0])
        data_max = torch.tensor([10.0, 20.0])
        transform = MinMaxTransform(data_min, data_max)
        d = transform.to_dict()
        restored = MinMaxTransform.from_dict(d)
        assert torch.allclose(restored.data_min, data_min)
        assert torch.allclose(restored.data_max, data_max)


class TestLogTransform:
    def test_log_forward(self) -> None:
        transform = LogTransform(offset=1.0)
        sample = torch.tensor([1.0, 2.0, 3.0])
        result = transform(sample)
        expected = torch.log(sample + 1.0)
        assert torch.allclose(result, expected)

    def test_log_inverse(self) -> None:
        transform = LogTransform(offset=1.0)
        sample = torch.tensor([1.0, 2.0, 3.0])
        result = transform.inverse(transform(sample))
        assert torch.allclose(result, sample, atol=1e-6)

    def test_log_serialization(self) -> None:
        transform = LogTransform(offset=0.5)
        d = transform.to_dict()
        assert d == {"type": "LogTransform", "offset": 0.5, "feature_dim": -1, "apply_to": None}
        restored = LogTransform.from_dict(d)
        assert restored.offset == 0.5


class TestComposedTransform:
    def test_composed_forward(self) -> None:
        t1 = TranslateScaleTransform(torch.tensor([1.0]), torch.tensor([2.0]))
        t2 = LogTransform(offset=0.0)
        transform = ComposedTransform([t1, t2])
        sample = torch.tensor([1.0, 2.0])
        result = transform(sample)
        expected = torch.log((sample + 1.0) * 2.0)
        assert torch.allclose(result, expected)

    def test_composed_inverse(self) -> None:
        t1 = TranslateScaleTransform(torch.tensor([1.0]), torch.tensor([2.0]))
        t2 = LogTransform(offset=0.0)
        transform = ComposedTransform([t1, t2])
        sample = torch.tensor([1.0, 2.0])
        result = transform.inverse(transform(sample))
        assert torch.allclose(result, sample, atol=1e-6)

    def test_composed_serialization(self) -> None:
        t1 = IdentityTransform()
        t2 = LogTransform(offset=1.0)
        transform = ComposedTransform([t1, t2])
        d = transform.to_dict()
        restored = ComposedTransform.from_dict(d)
        assert len(restored.transforms) == 2
        assert isinstance(restored.transforms[0], IdentityTransform)
        assert isinstance(restored.transforms[1], LogTransform)
        
        
class TestNestedTransform:
    
    def test_nested_transform_forward(self) -> None:
        
        t_dict = {
            'a': IdentityTransform(),
            'b': {
                'b1': IdentityTransform(),
                'b2': TranslateScaleTransform(torch.tensor([1.0]), torch.tensor([2.0]))
            }
        }
        
        t = NestedTransform(t_dict)
        
        sample = {
            'a': torch.tensor([-1.0, -2.0]),
            'b': {
                'b1': torch.tensor([42.0, 123.0]),
                'b2': torch.tensor([1.0, 2.0])
            }
        }
        out = t(sample)
        
        
        out_inv = t.inverse(out)
        
        def dicts_equal(d1, d2) -> bool:
            if d1.keys() != d2.keys():
                return False
            for k in d1:
                v1, v2 = d1[k], d2[k]
                if isinstance(v1, dict) and isinstance(v2, dict):
                    if not dicts_equal(v1, v2):
                        return False
                else:
                    if (v1 != v2).any():
                        return False
            return True
        
        assert dicts_equal(sample, out_inv)
        


class TestYAMLSerialization:
    def test_yaml_roundtrip(self) -> None:
        transform = StandardNormalizeTransform(
            mean=torch.tensor([1.0, 2.0]),
            std=torch.tensor([0.5, 1.0])
        )
        yaml_str = transform.to_yaml()
        restored = Transform.from_yaml(yaml_str)
        assert isinstance(restored, StandardNormalizeTransform)
        assert torch.allclose(restored.mean, transform.mean)

    def test_transform_dict_yaml(self) -> None:
        transforms = {
            "norm": StandardNormalizeTransform(torch.tensor([0.0]), torch.tensor([1.0])),
            "log": LogTransform(offset=1.0),
        }
        yaml_str = transform_dict_to_yaml(transforms)
        restored = transform_dict_from_yaml(yaml_str)
        assert "norm" in restored
        assert "log" in restored
        assert isinstance(restored["norm"], StandardNormalizeTransform)
        assert isinstance(restored["log"], LogTransform)

    def test_unknown_transform_type(self) -> None:
        d = {"type": "UnknownTransform"}
        with pytest.raises(ValueError, match="Unknown transform type"):
            Transform.from_dict(d)


class TestRegistry:
    def test_registry_contains_transforms(self) -> None:
        assert "IdentityTransform" in TRANSFORM_REGISTRY
        assert "TranslateScaleTransform" in TRANSFORM_REGISTRY
        assert "StandardNormalizeTransform" in TRANSFORM_REGISTRY
        assert "MinMaxTransform" in TRANSFORM_REGISTRY
        assert "LogTransform" in TRANSFORM_REGISTRY
        assert "ComposedTransform" in TRANSFORM_REGISTRY


class TestTransformBaseContract:
    def test_base_input_support_raises_not_implemented(self) -> None:
        transform = Transform()

        with pytest.raises(NotImplementedError, match="input_support"):
            _ = transform.input_support

    def test_base_transform_support_raises_not_implemented(self) -> None:
        transform = Transform()
        support = VariableSupport(
            lower_bounds=torch.tensor([-1.0]),
            upper_bounds=torch.tensor([1.0]),
            lower_open=torch.tensor([False]),
            upper_open=torch.tensor([False]),
        )

        with pytest.raises(NotImplementedError, match="transform_support"):
            transform.transform_support(support)


class TestIdentityTransformSupport:
    def test_identity_transform_support_preserves_input_support(self) -> None:
        transform = IdentityTransform()
        support = VariableSupport(
            lower_bounds=torch.tensor([-1.0, 0.0]),
            upper_bounds=torch.tensor([1.0, 10.0]),
            lower_open=torch.tensor([False, False]),
            upper_open=torch.tensor([False, False]),
        )

        transformed = transform.transform_support(support)

        assert transformed is support
        assert torch.equal(transformed.lower_bounds, support.lower_bounds)
        assert torch.equal(transformed.upper_bounds, support.upper_bounds)
        assert torch.equal(transformed.lower_open, support.lower_open)
        assert torch.equal(transformed.upper_open, support.upper_open)


class TestTranslateScaleTransformSupport:
    def test_positive_scale_maps_bounds_and_preserves_open_flags(self) -> None:
        transform = TranslateScaleTransform(
            translate=torch.tensor([1.0, -2.0]),
            scale=torch.tensor([2.0, 0.5]),
        )
        support = VariableSupport(
            lower_bounds=torch.tensor([0.0, -4.0]),
            upper_bounds=torch.tensor([2.0, 0.0]),
            lower_open=torch.tensor([False, True]),
            upper_open=torch.tensor([True, False]),
        )

        transformed = transform.transform_support(support)

        assert torch.allclose(transformed.lower_bounds, torch.tensor([2.0, -3.0]))
        assert torch.allclose(transformed.upper_bounds, torch.tensor([6.0, -1.0]))
        assert torch.equal(transformed.lower_open, support.lower_open)
        assert torch.equal(transformed.upper_open, support.upper_open)

    def test_negative_scale_flips_bounds_and_swaps_open_flags(self) -> None:
        transform = TranslateScaleTransform(
            translate=torch.tensor([0.0, 1.0]),
            scale=torch.tensor([-2.0, 3.0]),
        )
        support = VariableSupport(
            lower_bounds=torch.tensor([-1.0, 0.0]),
            upper_bounds=torch.tensor([4.0, 2.0]),
            lower_open=torch.tensor([False, True]),
            upper_open=torch.tensor([True, False]),
        )

        transformed = transform.transform_support(support)

        assert torch.allclose(transformed.lower_bounds, torch.tensor([-8.0, 3.0]))
        assert torch.allclose(transformed.upper_bounds, torch.tensor([2.0, 9.0]))
        assert torch.equal(transformed.lower_open, torch.tensor([True, True]))
        assert torch.equal(transformed.upper_open, torch.tensor([False, False]))

    def test_zero_scale_raises_value_error(self) -> None:
        transform = TranslateScaleTransform(
            translate=torch.tensor([0.0, 1.0]),
            scale=torch.tensor([1.0, 0.0]),
        )
        support = VariableSupport(
            lower_bounds=torch.tensor([-1.0, 0.0]),
            upper_bounds=torch.tensor([1.0, 2.0]),
            lower_open=torch.tensor([False, False]),
            upper_open=torch.tensor([False, False]),
        )

        with pytest.raises(ValueError, match="scale"):
            transform.transform_support(support)

    def test_support_variable_shape_mismatch_raises_value_error(self) -> None:
        transform = TranslateScaleTransform(
            translate=torch.tensor([0.0, 1.0]),
            scale=torch.tensor([1.0, 2.0]),
        )
        support = VariableSupport(
            lower_bounds=torch.tensor([-1.0]),
            upper_bounds=torch.tensor([1.0]),
            lower_open=torch.tensor([False]),
            upper_open=torch.tensor([False]),
        )

        with pytest.raises(ValueError, match="variables"):
            transform.transform_support(support)

    def test_input_support_shape(self) -> None:
        transform = TranslateScaleTransform(
            translate=torch.tensor([1.0, 2.0, 3.0]),
            scale=torch.tensor([0.5, 1.0, 2.0]),
        )
        support = transform.input_support
        assert support.lower_bounds.shape == (3,)
        assert support.upper_bounds.shape == (3,)
        assert support.lower_open.shape == (3,)
        assert support.upper_open.shape == (3,)

    def test_input_support_inf_bounds_and_open_flags(self) -> None:
        transform = TranslateScaleTransform(
            translate=torch.tensor([0.0, 0.0]),
            scale=torch.tensor([1.0, -1.0]),
        )
        support = transform.input_support
        assert torch.all(support.lower_bounds == float("-inf"))
        assert torch.all(support.upper_bounds == float("inf"))
        assert torch.all(support.lower_open)
        assert torch.all(support.upper_open)

    def test_input_support_dtype_and_device_match_scale(self) -> None:
        translate = torch.tensor([1.0, 2.0], dtype=torch.float64)
        scale = torch.tensor([3.0, 4.0], dtype=torch.float64)
        transform = TranslateScaleTransform(translate=translate, scale=scale)
        support = transform.input_support
        assert support.lower_bounds.dtype == torch.float64
        assert support.upper_bounds.dtype == torch.float64
        assert support.lower_bounds.device == scale.device
        assert support.upper_bounds.device == scale.device

    def test_constructor_raises_on_translate_scale_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="translate"):
            TranslateScaleTransform(
                translate=torch.tensor([1.0]),
                scale=torch.tensor([1.0, 2.0]),
            )
