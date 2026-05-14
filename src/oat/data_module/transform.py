import torch
import torch.nn as nn
import yaml
from typing import Any, cast

from oat.data_module.support import VariableSupport

TRANSFORM_REGISTRY: dict[str, type['Transform']] = {}

def register_transform(cls: type['Transform']) -> type['Transform']:
    TRANSFORM_REGISTRY[cls.__name__] = cls
    return cls

class Transform(nn.Module):
    """Base class for data transformations."""

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Transform forward method not implemented.")

    def inverse(self, sample: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Transform inverse method not implemented.")

    @property
    def input_support(self) -> "VariableSupport":
        raise NotImplementedError("Transform input_support property not implemented.")

    def transform_support(self, support: "VariableSupport") -> "VariableSupport":
        raise NotImplementedError("Transform transform_support method not implemented.")
    
    # -------- YAML serialization --------
    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation (override in subclasses)."""
        return {"type": self.__class__.__name__}
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'Transform':
        """Reconstruct object from dictionary (dispatches to subclass registry)."""
        ttype = d["type"]
        if ttype not in TRANSFORM_REGISTRY:
            raise ValueError(f"Unknown transform type: {ttype}")
        return TRANSFORM_REGISTRY[ttype].from_dict(d)

    # YAML entrypoints
    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict())

    @staticmethod
    def from_yaml(yaml_str: str) -> 'Transform':
        d = yaml.safe_load(yaml_str)
        return Transform.from_dict(d)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
    
@register_transform
class IdentityTransform(Transform):
    """A no-op transformation that returns the input unchanged."""

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """Return the sample unchanged."""
        return sample

    def inverse(self, sample: torch.Tensor) -> torch.Tensor:
        """Return the sample unchanged."""
        return sample

    def transform_support(self, support: "VariableSupport") -> "VariableSupport":
        return support
    
    # -------- YAML serialization --------
    def to_dict(self) -> dict[str, Any]:
        return {"type": "IdentityTransform"}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'IdentityTransform':
        return cls()
    
    
@register_transform
class TranslateScaleTransform(Transform):
    """Applies translation and scaling to the input data."""

    def __init__(self, translate: torch.Tensor, scale: torch.Tensor, channel_dim: int = -1):
        """
        Args:
            translate (tuple): Translation values for each dimension.
            scale (tuple): Scaling factors for each dimension.
        """
        super().__init__()
        if translate.numel() != scale.numel():
            raise ValueError(
                f"translate and scale must have the same number of elements, "
                f"got {translate.numel()} and {scale.numel()}."
            )
        self.register_buffer("translate", translate)
        self.register_buffer("scale", scale)
        self.channel_dim = channel_dim

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply translation and scaling to the sample."""

        # Align translate/scale shape with sample shape
        shape = [1] * sample.dim()
        shape[self.channel_dim] = -1
        translate = self.translate.view(shape)
        scale = self.scale.view(shape)
        return (sample + translate) * scale
        

    def inverse(self, sample: torch.Tensor) -> torch.Tensor:
        """Revert the translation and scaling applied to the sample."""

        # Align translate/scale shape with sample shape
        shape = [1] * sample.dim()
        shape[self.channel_dim] = -1
        translate = self.translate.view(shape)
        scale = self.scale.view(shape)
        return sample / scale - translate

    @property
    def input_support(self) -> "VariableSupport":
        n_variables = int(self.translate.numel())
        return VariableSupport(
            lower_bounds=torch.full(
                (n_variables,),
                float("-inf"),
                dtype=self.scale.dtype,
                device=self.scale.device,
            ),
            upper_bounds=torch.full(
                (n_variables,),
                float("inf"),
                dtype=self.scale.dtype,
                device=self.scale.device,
            ),
            lower_open=torch.ones(n_variables, dtype=torch.bool, device=self.scale.device),
            upper_open=torch.ones(n_variables, dtype=torch.bool, device=self.scale.device),
        )

    def transform_support(self, support: "VariableSupport") -> "VariableSupport":
        n_variables = int(self.translate.numel())
        n_support_variables = int(support.lower_bounds.numel())
        if n_support_variables != n_variables:
            raise ValueError(
                "Mismatch in variables between support and transform: "
                f"support has {n_support_variables} variables but transform has {n_variables} variables."
            )

        if torch.any(self.scale == 0):
            raise ValueError("scale must be nonzero for support transformation.")

        neg = self.scale < 0
        lb_raw = (support.lower_bounds + self.translate) * self.scale
        ub_raw = (support.upper_bounds + self.translate) * self.scale
        lower_bounds = torch.where(neg, ub_raw, lb_raw)
        upper_bounds = torch.where(neg, lb_raw, ub_raw)
        lower_open = torch.where(neg, support.upper_open, support.lower_open)
        upper_open = torch.where(neg, support.lower_open, support.upper_open)

        return VariableSupport(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            lower_open=lower_open,
            upper_open=upper_open,
        )

    # -------- YAML serialization --------
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "TranslateScaleTransform",
            "translate": self.translate.tolist(),
            "scale": self.scale.tolist(),
            "channel_dim": self.channel_dim
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'TranslateScaleTransform':
        return cls(
            translate=torch.tensor(d["translate"]),
            scale=torch.tensor(d["scale"]),
            channel_dim=d.get("channel_dim", -1) 
        )
    
@register_transform
class StandardNormalizeTransform(TranslateScaleTransform):
    """Normalizes data to have zero mean and unit variance."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, channel_dim: int = -1):
        """
        Args:
            mean (tuple): Mean values for each dimension.
            std (tuple): Standard deviation values for each dimension.
        """
        super().__init__(translate=-mean, scale=1./std, channel_dim=channel_dim)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "StandardNormalizeTransform",
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "channel_dim": self.channel_dim
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'StandardNormalizeTransform':
        return cls(
            mean=torch.tensor(d["mean"]),
            std=torch.tensor(d["std"]),
            channel_dim=d.get("channel_dim", -1)
        )
        
@register_transform
class MinMaxTransform(TranslateScaleTransform):
    """Scales data to a specified range using min-max normalization."""

    def __init__(self, data_min: torch.Tensor, data_max: torch.Tensor, channel_dim: int = -1):
        """
        Args:
            data_min (tuple): Minimum values for each dimension.
            data_max (tuple): Maximum values for each dimension.
            feature_range (tuple): Desired range of transformed data.
        """

        scale = 1 / (data_max - data_min)
        translate = -data_min
        super().__init__(translate=translate, scale=scale, channel_dim=channel_dim)
        
        self.register_buffer("data_min", data_min)
        self.register_buffer("data_max", data_max)
        
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "MinMaxTransform",
            "data_min": self.data_min.tolist(),
            "data_max": self.data_max.tolist(),
            "channel_dim": self.channel_dim
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'MinMaxTransform':
        return cls(
            data_min=torch.tensor(d["data_min"]),
            data_max=torch.tensor(d["data_max"]),
            channel_dim=d.get("channel_dim", -1)
        )
        
@register_transform
class LogTransform(Transform):
    """Applies logarithmic transformation to the input data."""

    def __init__(self, 
                 offset: float = 1e-4, 
                 feature_dim: int = -1,
                 apply_to: list[int] | None = None) -> None:
        """
        Args:
            offset (float): Small constant to avoid log(0).
            feature_dim (int): Dimension of features in the input tensor.
            apply_to (list[int] | None): Indices of features to apply the transform to. If None, applies to all features.
        """
        super().__init__()
        
        # TODO should we make EVERY attribute a buffer?
        self.offset = offset
        self.feature_dim = feature_dim
        
        # Convert to tensor once during initialization if provided
        if apply_to is not None:
            self.register_buffer("apply_to", torch.tensor(apply_to, dtype=torch.long))
            # Precompute boolean mask for efficient indexing
            self._use_mask = True
        else:
            self.apply_to = None
            self._use_mask = False
            
    def _forward_transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + self.offset)
    
    def _inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x) - self.offset

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply logarithmic transformation to the sample."""
        if not self._use_mask:
            return self._forward_transform(sample)
        
        # Clone to avoid modifying input
        result = sample.clone()
        
        # Use advanced indexing for efficient in-place update
        indices = self.apply_to
        transformed = result.index_select(self.feature_dim, indices).copy_(
            self._forward_transform(sample.index_select(self.feature_dim, indices))
        )
        result.index_copy_(self.feature_dim, indices, transformed)
        
        return result

    def inverse(self, sample: torch.Tensor) -> torch.Tensor:
        """Revert the logarithmic transformation applied to the sample."""
        if not self._use_mask:
            return self._inverse_transform(sample)
        
        # Clone to avoid modifying input
        result = sample.clone()
        
        # Use advanced indexing for efficient in-place update
        indices = self.apply_to
        transformed = result.index_select(self.feature_dim, indices).copy_(
            self._inverse_transform(sample.index_select(self.feature_dim, indices))
        )
        result.index_copy_(self.feature_dim, indices, transformed)
        
        return result
    
    def to_dict(self) -> dict[str, Any]:
        apply_to_list = self.apply_to.tolist() if self.apply_to is not None else None
        return {
            "type": "LogTransform", 
            "offset": float(self.offset), 
            "feature_dim": self.feature_dim, 
            "apply_to": apply_to_list
        }

    @property
    def input_support(self) -> "VariableSupport":
        return VariableSupport(
            lower_bounds=torch.tensor([-self.offset]),
            upper_bounds=torch.tensor([float("inf")]),
            lower_open=torch.tensor([True]),
            upper_open=torch.tensor([True]),
        )

    def transform_support(self, support: "VariableSupport") -> "VariableSupport":
        if self.apply_to is None:
            if not support.is_subset(self.input_support):
                raise ValueError(
                    f"Support is not a subset of the input support (x > {-self.offset}). "
                    "All variables must lie strictly above the negative offset."
                )
            return VariableSupport(
                lower_bounds=torch.log(support.lower_bounds + self.offset),
                upper_bounds=torch.log(support.upper_bounds + self.offset),
                lower_open=support.lower_open.clone(),
                upper_open=support.upper_open.clone(),
            )
        else:
            indices = self.apply_to
            sub_support = VariableSupport(
                lower_bounds=support.lower_bounds[indices],
                upper_bounds=support.upper_bounds[indices],
                lower_open=support.lower_open[indices],
                upper_open=support.upper_open[indices],
            )
            if not sub_support.is_subset(self.input_support):
                raise ValueError(
                    f"Support at selected indices is not a subset of the input support (x > {-self.offset}). "
                    "Selected variables must lie strictly above the negative offset."
                )
            new_lb = support.lower_bounds.clone()
            new_ub = support.upper_bounds.clone()
            new_lb[indices] = torch.log(support.lower_bounds[indices] + self.offset)
            new_ub[indices] = torch.log(support.upper_bounds[indices] + self.offset)
            return VariableSupport(
                lower_bounds=new_lb,
                upper_bounds=new_ub,
                lower_open=support.lower_open.clone(),
                upper_open=support.upper_open.clone(),
            )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'LogTransform':
        return cls(
            offset=d["offset"], 
            feature_dim=d.get("feature_dim", -1), 
            apply_to=d.get("apply_to", None)
        )

@register_transform    
class ComposedTransform(Transform):
    """Concatenates multiple transformations."""

    def __init__(self, transforms: list[Transform]) -> None:
        """
        Args:
            transforms (list): List of Transform instances to concatenate.
        """
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply all transformations in sequence to the sample."""
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def inverse(self, sample: torch.Tensor) -> torch.Tensor:
        """Revert all transformations in reverse sequence applied to the sample."""
        for transform in reversed(self.transforms):
            sample = transform.inverse(sample)
        return sample

    def transform_support(self, support: "VariableSupport") -> "VariableSupport":
        for transform in self.transforms:
            support = transform.transform_support(support)
        return support
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ComposedTransform",
            "transforms": [t.to_dict() for t in self.transforms]
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'ComposedTransform':
        transforms = [Transform.from_dict(td) for td in d["transforms"]]
        return cls(transforms)
    
    def __repr__(self) -> str:
        chain = " -> ".join(t.__class__.__name__ for t in self.transforms)
        return f"{self.__class__.__name__}({chain})"
    
class NestedTransform(Transform):
    """Applies transforms to a nested dictionary of tensors.
    
    The transform_dict should mirror the structure of the data dictionary.
    Values can be either Transform instances or nested dictionaries.
    If a key is missing in transform_dict, no transform is applied to that data key.
    """

    def __init__(self, transform_dict: dict[str, Transform | dict]) -> None:
        """
        Args:
            transform_dict: Dictionary with the same structure as the data.
                           Values are either Transform instances or nested dicts of transforms.
        """
        super().__init__()
        self.transform_dict = nn.ModuleDict(self._build(transform_dict))
        
    def _build(self, d: dict[str, Transform | dict]) -> dict[str, Transform]:
        built = {}
        
        for k,v in d.items():
            if isinstance(v, Transform):
                built[k] = v
            elif isinstance(v, dict):
                built[k] = NestedTransform(v)
            else:
                raise TypeError()
        return built

    def forward(self, sample: dict[str, torch.Tensor | dict]) -> dict[str, torch.Tensor | dict]:
        """Apply transforms to the nested dictionary."""
        # return self._apply_recursive(sample, self.transform_dict)
        
        out = {}
        for k,v in sample.items():
            if k in self.transform_dict:
                out[k] = self.transform_dict[k](v)
            else:
                out[k] = v
        return out

    def inverse(self, sample: dict[str, torch.Tensor | dict]) -> dict[str, torch.Tensor | dict]:
        """Apply inverse transforms to the nested dictionary."""
        # return self._apply_inverse_recursive(sample, self.transform_dict)
        
        out = {}
        for k,v in sample.items():
            if k in self.transform_dict:
                out[k] = self.transform_dict[k].inverse(v)
            else:
                out[k] = v
        return out

    def transform_support(
        self,
        support_dict: dict[str, VariableSupport | dict | None],
    ) -> dict[str, VariableSupport | dict | None]:
        out: dict[str, VariableSupport | dict | None] = {}
        for key, value in support_dict.items():
            if value is None:
                out[key] = None
            elif key not in self.transform_dict:
                out[key] = value
            else:
                transform = self.transform_dict[key]
                if isinstance(transform, NestedTransform):
                    if not isinstance(value, dict):
                        raise TypeError(
                            f"Support for key '{key}' must be a dict because the "
                            "transform is nested."
                        )
                    out[key] = transform.transform_support(value)
                else:
                    if isinstance(value, dict):
                        raise TypeError(
                            f"Support for key '{key}' must be a VariableSupport "
                            "because the transform is a leaf transform."
                        )
                    out[key] = transform.transform_support(cast(VariableSupport, value))
        return out

    def _dict_to_serializable(self, d: dict) -> dict:
        """Convert nested dict of transforms to serializable format."""
        #TODO check
        result = {}
        for key, value in d.items():
            if isinstance(value, Transform):
                result[key] = value.to_dict()
            elif isinstance(value, dict):
                result[key] = self._dict_to_serializable(value)
            else:
                raise ValueError(f"Invalid value type in transform_dict: {type(value)}")
        return result

    def _dict_from_serializable(self, d: dict) -> dict:
        """Convert serializable format back to nested dict of transforms."""
        #TODO check
        result = {}
        for key, value in d.items():
            if isinstance(value, dict) and "type" in value:
                # This is a serialized transform
                result[key] = Transform.from_dict(value)
            elif isinstance(value, dict):
                # This is a nested dictionary
                result[key] = self._dict_from_serializable(value)
            else:
                raise ValueError(f"Invalid value in serialized transform_dict: {value}")
        return result

    def to_dict(self) -> dict[str, Any]:
        # TODO check
        return {
            "type": "NestedTransform",
            "transform_dict": self._dict_to_serializable(self.transform_dict)
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'NestedTransform':
        #TODO check
        instance = cls({})
        instance.transform_dict = instance._dict_from_serializable(d["transform_dict"])
        return instance
    
    def __repr__(self) -> str:
        def render(td: nn.ModuleDict, indent: int = 0) -> list[str]:
            lines = []
            pad = "  " * indent
            for key, value in td.items():
                if isinstance(value, NestedTransform):
                    lines.append(f"{pad}{key}:")
                    lines.extend(render(value.transform_dict, indent + 1))
                else:
                    # leaf Transform
                    lines.append(f"{pad}{key}: {repr(value)}")
            return lines

        body = "\n".join(render(self.transform_dict))
        return f"{self.__class__.__name__}(\n{body}\n)"

def transform_dict_to_yaml(transform_dict: dict[str, Transform]) -> str:
    transform_dicts = {key: tr.to_dict() for key, tr in transform_dict.items()}
    td = {'transforms': transform_dicts}
    return yaml.safe_dump(td)

def transform_dict_from_yaml(yaml_str: str) -> dict[str, Transform]:
    d = yaml.safe_load(yaml_str)
    transform_dict = {}
    for key, tr_dict in d['transforms'].items():
        transform_dict[key] = Transform.from_dict(tr_dict)
    return transform_dict

