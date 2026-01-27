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
            
        mask = torch.zeros(
            (coords_1.shape[0], coords_1.shape[1], coords_2.shape[1]),
            dtype=torch.bool,
            device=coords_1.device,
        )
        for b in range(coords_1.shape[0]):
            
            if self.mode == 'radius':
                edge_index = radius(
                    x=coords_1[b],
                    y=coords_2[b],
                    r=self.radius,
                    max_num_neighbors=self.max_num_neighbors,
                )
            elif self.mode == 'knn':
                edge_index = knn(
                    x=coords_1[b],
                    y=coords_2[b],
                    k=self.k,
                )
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
            
            mask[b][edge_index[1], edge_index[0]] = True
        
        return mask  # [B, N_1, N_2]
        
        