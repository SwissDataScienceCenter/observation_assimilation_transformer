from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import math

class PreOATLossWeightScheduler(ABC, nn.Module):
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
        self.register_buffer("current_step", torch.tensor(0.0, dtype=torch.long))
        self.register_buffer("current_weight", torch.tensor(start_weight, dtype=torch.float32))

    def forward(self):
        return self.current_weight.item()
    
    @abstractmethod
    def step(self):
        '''Not implemented'''
        pass
    
    def reset(self):
        self.current_step.fill_(torch.tensor(0.0, dtype=torch.long))
        self.current_weight.fill_(torch.tensor(self.start_weight, dtype=torch.float32))


class LinearPreOATLossWeightScheduler(PreOATLossWeightScheduler):
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