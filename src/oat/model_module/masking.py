"""
Geometry-Aware Multi-Head Attention using Direct Indices

Simplest implementation: just use neighbor indices from torch_cluster.
No SparseTensor abstraction - just gather operations with indices.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal
from torch_cluster import knn, radius


class SpatialNeighborSearch(nn.Module):
    """
    Efficient spatial neighbor search using torch_cluster.
    
    Returns neighbor indices directly (no SparseTensor).
    """
    
    def __init__(
        self,
        mode: Literal['radius', 'knn'] = 'knn',
        radius: Optional[float] = None,
        k: Optional[int] = None,
        max_num_neighbors: int = 32,
    ):
        """
        Args:
            mode: 'radius' or 'knn'
            radius: Distance threshold for radius mode
            k: Number of neighbors for knn mode
            max_num_neighbors: Max neighbors per point in radius mode
        """
        super().__init__()
        
        self.mode = mode
        self.max_num_neighbors = max_num_neighbors
        
        if mode == 'radius':
            if radius is None:
                raise ValueError("radius must be specified for mode='radius'")
            self.radius = radius
        elif mode == 'knn':
            if k is None:
                raise ValueError("k must be specified for mode='knn'")
            self.k = k
        else:
            raise ValueError(f"mode must be 'radius' or 'knn', got {mode}")
    
    def forward(
        self,
        coords_1: torch.Tensor,
        coords_2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute binary mask based on spatial neighbors.
        
        Args:
            coords_1: [B, N_1, D] first set of coordinates
            coords_2: [B, N_2, D] second set of coordinates
        
        Returns:
            mask: [B, N_1, N_2] binary mask indicating neighbors
        """
        
        
        
        if coords_2 is None:
            coords_2 = coords_1

        B, N1, _ = coords_1.shape
        _, N2, _ = coords_2.shape

        # Flatten coordinates across batch dimension for batched neighbor search
        coords_1_flat = coords_1.reshape(B * N1, -1)
        coords_2_flat = coords_2.reshape(B * N2, -1)

        # Batch indices for each point in the flattened tensors
        batch_x = torch.arange(B, device=coords_1.device).repeat_interleave(N1)
        batch_y = torch.arange(B, device=coords_1.device).repeat_interleave(N2)

        # Perform neighbor search in a single batched call
        if self.mode == 'radius':
            edge_index = radius(
                x=coords_1_flat,
                y=coords_2_flat,
                r=self.radius,
                batch_x=batch_x,
                batch_y=batch_y,
                max_num_neighbors=self.max_num_neighbors,
            )
        elif self.mode == 'knn':
            edge_index = knn(
                x=coords_1_flat,
                y=coords_2_flat,
                k=self.k,
                batch_x=batch_x,
                batch_y=batch_y,
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # Initialize dense mask [B, N1, N2]
        mask = torch.zeros(
            (B, N1, N2),
            dtype=torch.bool,
            device=coords_1.device,
        )

        # edge_index[1] indexes coords_1_flat (source), edge_index[0] indexes coords_2_flat (target)
        src_flat = edge_index[1]
        tgt_flat = edge_index[0]

        # Recover batch indices from flattened indices
        b_idx = batch_x[src_flat]

        # Convert flattened indices to per-batch indices
        i1 = src_flat - b_idx * N1  # indices in coords_1 within each batch
        i2 = tgt_flat - b_idx * N2  # indices in coords_2 within each batch

        # Fill mask in a vectorized way
        mask[b_idx, i1, i2] = True
        return mask  # [B, N_1, N_2]
        
        