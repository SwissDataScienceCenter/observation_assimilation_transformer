import torch
import yaml
from typing import Any, Callable

TRANSFORM_REGISTRY: dict[str, type['Transform']] = {}

def register_transform(cls: type['Transform']) -> type['Transform']:
    TRANSFORM_REGISTRY[cls.__name__] = cls
    return cls

class Transform(Callable[[torch.Tensor], torch.Tensor]):
    """Base class for data transformations."""

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Transform __call__ method not implemented.")

    def inverse(self, sample: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Transform inverse method not implemented.")
    
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
    
    
@register_transform
class IdentityTransform(Transform):
    """A no-op transformation that returns the input unchanged."""

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """Return the sample unchanged."""
        return sample

    def inverse(self, sample: torch.Tensor) -> torch.Tensor:
        """Return the sample unchanged."""
        return sample
    
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
        self.translate = translate
        self.scale = scale
        self.channel_dim = channel_dim

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply translation and scaling to the sample."""
        translate = self.translate.to(sample.device)
        scale = self.scale.to(sample.device)
        # Align translate/scale shape with sample shape
        shape = [1] * sample.dim()
        shape[self.channel_dim] = -1
        translate = translate.view(shape)
        scale = scale.view(shape)
        return (sample + translate) * scale
        

    def inverse(self, sample: torch.Tensor) -> torch.Tensor:
        """Revert the translation and scaling applied to the sample."""
        translate = self.translate.to(sample.device)
        scale = self.scale.to(sample.device)
        # Align translate/scale shape with sample shape
        shape = [1] * sample.dim()
        shape[self.channel_dim] = -1
        translate = translate.view(shape)
        scale = scale.view(shape)
        return sample / scale - translate

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
        self.mean = mean
        self.std = std
        super().__init__(translate=-mean, scale=1./std, channel_dim=channel_dim)
        
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
        self.data_min = data_min
        self.data_max = data_max
        scale = 1 / (data_max - data_min)
        translate = -data_min
        super().__init__(translate=translate, scale=scale, channel_dim=channel_dim)
        
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

    def __init__(self, offset: float = 0.):
        """
        Args:
            offset (float): Small constant to avoid log(0).
        """
        self.offset = offset

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply logarithmic transformation to the sample."""
        return torch.log(sample + self.offset)

    def inverse(self, sample: torch.Tensor) -> torch.Tensor:
        """Revert the logarithmic transformation applied to the sample."""
        return torch.exp(sample) - self.offset
    
    def to_dict(self) -> dict[str, Any]:
        return {"type": "LogTransform", "offset": float(self.offset)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'LogTransform':
        return cls(offset=d["offset"])


@register_transform    
class ComposedTransform(Transform):
    """Concatenates multiple transformations."""

    def __init__(self, transforms: list[Transform]):
        """
        Args:
            transforms (list): List of Transform instances to concatenate.
        """
        self.transforms = transforms

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply all transformations in sequence to the sample."""
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def inverse(self, sample: torch.Tensor) -> torch.Tensor:
        """Revert all transformations in reverse sequence applied to the sample."""
        for transform in reversed(self.transforms):
            sample = transform.inverse(sample)
        return sample
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ComposedTransform",
            "transforms": [t.to_dict() for t in self.transforms]
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'ComposedTransform':
        transforms = [Transform.from_dict(td) for td in d["transforms"]]
        return cls(transforms)
    
    
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
