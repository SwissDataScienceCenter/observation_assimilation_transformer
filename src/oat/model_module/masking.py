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
    
    Computes and returns a dense boolean neighbor mask (no SparseTensor),
    using neighbor indices from torch_cluster internally.
    """
    
    def __init__(
        self,
        mode: Literal['radius', 'knn', 'full'] = 'knn',
        radius: Optional[float] = None,
        k: Optional[int] = None,
        max_num_neighbors: int = 32,
    ) -> None:
        """
        Args:
            mode: 'radius', 'knn', or 'full'
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
        elif mode == 'full':
            pass  # no additional parameters needed
        else:
            raise ValueError(f"mode must be 'radius', 'knn', or 'full', got {mode}")
    
    def forward(
        self,
        coords_1: torch.Tensor,
        coords_2: Optional[torch.Tensor] = None,
        return_full_mask: bool = False,
    ) -> Optional[torch.Tensor]:
        """
        Compute edges of flattened batch point clouds and return edges

        Args:
            coords_1: [B, N_1, D] first set of coordinates
            coords_2: [B, N_2, D] second set of coordinates (if None, use coords_1 for both)
            return_full_mask: If True, return a dense boolean mask of shape [B, N_1, N_2] instead of edge indices.
        Returns:
            In mode 'full': None, whatever return_full_mask. Every point is a
            neighbor of every other one, which the attention treats as dense
            and unmasked.
            Otherwise, if return_full_mask is False:
                edge_index: [2, E] edge indices in flattened batch space
            Otherwise, if return_full_mask is True:
                mask: [B, N_1, N_2] boolean mask where True indicates a neighbor relationship between points in coords_1 and coords_2.
        """
        if coords_2 is None:
            coords_2 = coords_1

        B, N1, _ = coords_1.shape
        _, N2, _ = coords_2.shape

        if self.mode == 'full':
            # Every point is a neighbor of every other one, so no edges are
            # built: the attention treats None as "attend to all" (dense,
            # unmasked).
            return None

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
        if return_full_mask:
            return self.create_full_mask_from_edges(edge_index, B, N1, N2)
        else:
            return edge_index  # [2, E] edge indices in flattened batch space
        
        
    def create_full_mask_from_edges(self, edge_index: torch.Tensor, B: int, N1: int, N2: int) -> torch.Tensor:
        """
        Create a dense boolean mask from edge indices.
        Args:
            edge_index: [2, E] edge indices in flattened batch space
            B: Batch size
            N1: Number of points in coords_1
            N2: Number of points in coords_2
        Returns:
            mask: [B, N1, N2] boolean mask where True indicates a neighbor relationship between points in coords_1 and coords_2.
        """
        # Create a dense boolean mask from edge_index
        mask = torch.zeros(
            (B, N1, N2),
            dtype=torch.bool,
        )
        batch_x = torch.arange(B).repeat_interleave(N1)
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