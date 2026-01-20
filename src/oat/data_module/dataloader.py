import torch
from typing import Any, Callable, Dict
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from .wind_dataset import WindDataset

class OATLoader(DataLoader[Dict[str, torch.Tensor]]):
    """
    A DataLoader that automatically:
       - moves batches to a device
       - casts to dtype
       - applies transforms
       
    Works exactly like DataLoader: accepts a dataset directly.
    """
    def __init__(
        self,
        dataset: WindDataset,
        device: str="cpu",
        dtype: torch.dtype=torch.float32,
        non_blocking: bool=True,
        transforms: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        **dataloader_kwargs : Any,
    ) -> None:
        super().__init__(dataset, **dataloader_kwargs)

        self.device = torch.device(device)
        self.dtype = dtype
        self.non_blocking = non_blocking
        self.transforms = transforms

    def __iter__(self) -> _BaseDataLoaderIter:
        """ Ensures post-processing is applied to each batch.
        """
        for batch in super().__iter__():
            yield self._process(batch, transform=self.transforms)

    # ---------------------------------------------------------
    # Recursive device + dtype + flatten
    # ---------------------------------------------------------
    def _process(self, data: torch.Tensor | dict[str, torch.Tensor], transform: dict | Callable | None) -> Any:
        """ Applies transforms to data and move to indicated device.

        Args:
            data: Tensor or dict containing relevant data.
            transform: Transform or dict thereof that needs to be applied to the data.

        Returns:
            Transformed data.
        """
        if isinstance(data, torch.Tensor):
            # Move to device
            if transform is not None:
                assert callable(transform), "Transform must be callable."
                data = transform(data)
                assert isinstance(data, torch.Tensor), "Transform must return a torch.Tensor."
            data = data.to(self.device, non_blocking=self.non_blocking)

            # Cast dtype
            if self.dtype is not None and data.dtype != self.dtype:
                data = data.to(self.dtype)

            return data

            
        elif isinstance(data, dict):
            if transform is None:
                return {k: self._process(v, None) for k, v in data.items()}
            else:
                assert isinstance(transform, dict), "Transform must be a dict if data is a dict."
                return {k: self._process(v, transform.get(k)) for k, v in data.items()}

        else:
            return data
        
        
@torch.no_grad()
def compute_stats_from_loader(
    loader: DataLoader[Dict[str, Any]],
    chan_dims: dict[str, Any],
    max_batches: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Compute mean/std/min/max for tensor leaves specified in `chan_dims`.

    - `chan_dims` mirrors the (nested) batch structure
    - Only tensor leaves with an associated channel dim are processed
    - Channel dims may be negative
    
    Args:
        loader: DataLoader providing batches of data.
        chan_dims: Nested dict specifying channel dims for tensor leaves.
        max_batches: Maximum number of batches to process. If None, process all.
        device: Device to perform computations on. If None, uses CPU.
        
    Returns:
        Dictionary of statistics for each tensor leaf.
    """

    stats: dict[tuple[str, ...], dict[str, torch.Tensor]] = {}

    def init_stats(shape: torch.Size) -> dict[str, torch.Tensor]:
        return dict(
            count=torch.zeros(shape, dtype=torch.float64, device=device),
            sum=torch.zeros(shape, dtype=torch.float64, device=device),
            sumsq=torch.zeros(shape, dtype=torch.float64, device=device),
            min=torch.full(shape, float("inf"), device=device),
            max=torch.full(shape, float("-inf"), device=device),
        )

    def update_tensor_stats(
        path: tuple[str, ...],
        t: torch.Tensor,
        channel_dim: int,
    ) -> None:
        """
        Update running stats for a single tensor leaf.
        """

        # normalize negative channel dim
        if channel_dim < 0:
            channel_dim = t.ndim + channel_dim

        # move channel dim to front, flatten everything else
        reduced = (
            t.swapdims(0, channel_dim)
             .reshape(t.shape[channel_dim], -1)
             .swapdims(0, 1)
        )

        stat_shape = reduced.shape[1:]

        if path not in stats:
            stats[path] = init_stats(stat_shape)

        s = stats[path]

        nan_mask = ~torch.isnan(reduced)
        safe = torch.where(
            nan_mask,
            reduced,
            torch.zeros_like(reduced, dtype=reduced.dtype, device=device),
        )

        s["sum"] += safe.sum(dim=0).double()
        s["sumsq"] += (safe ** 2).sum(dim=0).double()
        s["count"] += nan_mask.sum(dim=0).double()

        masked_min = torch.where(nan_mask, reduced, torch.inf)
        s["min"] = torch.minimum(s["min"], masked_min.amin(dim=0))

        masked_max = torch.where(nan_mask, reduced, -torch.inf)
        s["max"] = torch.maximum(s["max"], masked_max.amax(dim=0))

    def traverse(
        batch_obj: Any,
        spec_obj: Any,
        path: tuple[str, ...],
    ) -> None:
        """
        Traverse batch and channel-dim spec simultaneously.
        """

        if isinstance(spec_obj, dict):
            # descend into dict
            if not isinstance(batch_obj, dict):
                raise TypeError(
                    f"Structure mismatch at {'.'.join(path)}: "
                    f"expected dict, got {type(batch_obj)}"
                )
            for k, v in spec_obj.items():
                if k in batch_obj:
                    traverse(batch_obj[k], v, path + (k,))
            return

        # leaf: spec_obj is expected to be an int channel dim
        if not torch.is_tensor(batch_obj):
            raise TypeError(
                f"Expected tensor at {'.'.join(path)}, got {type(batch_obj)}"
            )

        if not isinstance(spec_obj, int):
            raise TypeError(
                f"Channel dim spec at {'.'.join(path)} must be int, got {type(spec_obj)}"
            )

        update_tensor_stats(path, batch_obj, spec_obj)

    # ---------------- main loop ----------------
    for i, batch in enumerate(loader):
        traverse(batch, chan_dims, path=())

        if max_batches is not None and i >= max_batches:
            break

    # ---------------- finalize -----------------
    def finalize(s: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mean = s["sum"] / s["count"]
        var = s["sumsq"] / s["count"] - mean ** 2
        std = torch.sqrt(torch.clamp(var, min=0.0))
        return dict(
            mean=mean.cpu(),
            std=std.cpu(),
            min=s["min"].cpu(),
            max=s["max"].cpu(),
        )

    final: dict[str, Any] = {}
    for path, s in stats.items():
        node = final
        for k in path[:-1]:
            node = node.setdefault(k, {})
        node[path[-1]] = finalize(s)

    return final