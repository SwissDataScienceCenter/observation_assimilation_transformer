from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np


import math
from typing import Optional

def estimate_total_steps(
    dataset_size: int,
    batch_size: int,
    max_epochs: Optional[int] = None,
    max_steps: Optional[int] = None,
    accumulate_grad_batches: int = 1,
    num_nodes: int = 1,
    num_gpus_per_node: int = 1,
    limit_train_batches: float | int = 1.0,
    drop_last: bool = False,
) -> int:
    """
    Estimate total optimizer steps similar to Lightning's `estimated_stepping_batches`,
    extended for multi-node + multi-GPU training.

    Args:
        dataset_size: total number of samples in dataset
        batch_size: batch size per GPU (per process)
        max_epochs: number of epochs
        max_steps: if set, overrides everything
        accumulate_grad_batches: gradient accumulation factor
        num_nodes: number of nodes
        num_gpus_per_node: GPUs per node
        limit_train_batches:
            float (0.0–1.0): fraction of batches
            int: absolute number of batches
        drop_last: whether dataloader drops last incomplete batch
    """

    # 1. max_steps overrides everything (Lightning behavior)
    if max_steps is not None and max_steps > 0:
        return max_steps

    assert max_epochs is not None, "Either max_epochs or max_steps must be set"

    # 2. Total number of processes (world size)
    world_size = num_nodes * num_gpus_per_node

    # 3. Effective global batch size
    global_batch_size = batch_size * world_size

    # 4. Batches per epoch (before limits)
    if drop_last:
        batches_per_epoch = dataset_size // global_batch_size
    else:
        batches_per_epoch = math.ceil(dataset_size / global_batch_size)

    # 5. Apply limit_train_batches
    if isinstance(limit_train_batches, float):
        batches_per_epoch = int(batches_per_epoch * limit_train_batches)
    else:
        batches_per_epoch = min(batches_per_epoch, limit_train_batches)

    # 6. Optimizer steps per epoch (after grad accumulation)
    steps_per_epoch = math.ceil(batches_per_epoch / accumulate_grad_batches)

    # 7. Total steps
    total_steps = steps_per_epoch * max_epochs

    return total_steps

class WeightScheduler(nn.Module, ABC):
    def __init__(self, start_weight, flat_warmup_steps, end_weight, total_steps):
        super().__init__()
        
        # TODO add docs

        assert total_steps > 0
        assert 0 <= flat_warmup_steps < total_steps
        
        self.start_weight = float(start_weight)
        self.end_weight = float(end_weight)
        self.total_steps = total_steps
        self.warmup_steps = flat_warmup_steps
        
        # These will automatically be saved in checkpoints
        # and are NOT trained
        self.register_buffer("current_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("current_weight", torch.tensor(start_weight, dtype=torch.float32))

    def forward(self):
        return self.current_weight.item()
    
    @abstractmethod
    def step(self):
        '''Not implemented'''
        pass
    
    def reset(self):
        self.current_step.zero_()
        self.current_weight.fill_(self.start_weight)


class LinearWeightScheduler(WeightScheduler):
    """
    Linearly schedules a loss weight after an initial flat warmup period.

    This scheduler keeps the weight fixed at `start_weight` for a specified
    number of warmup steps. After the warmup phase, the weight evolves
    linearly from `start_weight` to `end_weight` over the remaining training
    steps until `total_steps` is reached.
    
    Args:
        start_weight:
            Initial loss weight (float between 0.0 and 1.0) used during the warmup phase and as the
            starting value for the linear schedule.
        flat_warmup_steps:
            Number of initial steps during which the weight remains constant
            at `start_weight`.
        end_weight:
            Final loss weight (float between 0.0 and 1.0) reached at `total_steps`.
        total_steps:
            Total number of scheduler steps, including the warmup phase.
    """
    def __init__(self, start_weight, flat_warmup_steps, end_weight, total_steps):
        super().__init__(start_weight=start_weight, flat_warmup_steps=flat_warmup_steps, end_weight=end_weight, total_steps=total_steps)
    
    def step(self):
        if self.current_step.item() < self.warmup_steps:
            w = self.start_weight
        else:
            
            progress_steps = self.total_steps - self.warmup_steps
            progress = min(
                    (self.current_step.item() - self.warmup_steps) / progress_steps,
                    1.0,
                )
            w = self.start_weight + progress * (self.end_weight - self.start_weight)
        
        self.current_weight.fill_(w)
        self.current_step += 1
        
        return self.current_weight.item()
        
    def __repr__(self):
        if self.warmup_steps > 0:
            schedule_type = (
                f"Flat warmup for {self.warmup_steps} steps at {self.start_weight}"
                f" then linear interpolation to {self.end_weight}"
            )
        else:
            schedule_type = (
                f"Linear decay from {self.start_weight} to {self.end_weight}"
            )

        return (
            f"{self.__class__.__name__}(\n"
            f"  schedule={schedule_type},\n"
            f"  total_steps={self.total_steps}\n"
            f")"
        )
        
class ExponentialWeightScheduler(WeightScheduler):
    def __init__(self, start_weight, flat_warmup_steps, end_weight, total_steps, sharpness=0.5):
        if start_weight <= 0:
            raise ValueError("start_weight must be > 0 for ExponentialWeightScheduler")
        if end_weight <= 0:
            raise ValueError("end_weight must be > 0 for ExponentialWeightScheduler")
        super().__init__(start_weight=start_weight, flat_warmup_steps=flat_warmup_steps, end_weight=end_weight, total_steps=total_steps)

        # NOTE currently this is hardcoded by the default param.
        # and is not yet controlled by the model's config.
        self.sharpness = sharpness

    def step(self):
        step = self.current_step.item()

        if step < self.warmup_steps:
            w = self.start_weight

        elif step > self.total_steps:
            w = self.end_weight

        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)

            shaped_progress = progress ** self.sharpness

            w = self.start_weight * np.exp(
                -np.log(self.start_weight / self.end_weight) * shaped_progress
            )

        self.current_weight.fill_(w)
        self.current_step += 1

        return self.current_weight.item()
    
    
    def __repr__(self):
        if self.warmup_steps > 0:
            schedule_type = (
                f"Flat warmup for {self.warmup_steps} steps at {self.start_weight}"
                f" then exponential decay to {self.end_weight} with sharpness {self.sharpness}"
            )
        else:
            schedule_type = (
                f"Exponential decay from {self.start_weight} to {self.end_weight} with sharpness {self.sharpness}"
            )

        return (
            f"{self.__class__.__name__}(\n"
            f"  schedule={schedule_type},\n"
            f"  total_steps={self.total_steps}\n"
            f")"
        )
