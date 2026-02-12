import pytest
import torch
from oat.model_module.masking import SpatialNeighborSearch

class TestSpatialNeighborSearchInit:
    """Tests for SpatialNeighborSearch.__init__"""
    
    def test_init_knn_mode_with_k(self) -> None:
        """Test initialization with knn mode and k parameter"""
        search = SpatialNeighborSearch(mode='knn', k=8)
        assert search.mode == 'knn'
        assert search.k == 8
        assert search.max_num_neighbors == 32
    
    def test_init_radius_mode_with_radius(self) -> None:
        """Test initialization with radius mode and radius parameter"""
        search = SpatialNeighborSearch(mode='radius', radius=1.5, max_num_neighbors=16)
        assert search.mode == 'radius'
        assert search.radius == 1.5
        assert search.max_num_neighbors == 16
    
    def test_init_knn_mode_without_k_raises_error(self) -> None:
        """Test that knn mode without k raises ValueError"""
        with pytest.raises(ValueError, match="k must be specified for mode='knn'"):
            SpatialNeighborSearch(mode='knn')
    
    def test_init_radius_mode_without_radius_raises_error(self) -> None:
        """Test that radius mode without radius raises ValueError"""
        with pytest.raises(ValueError, match="radius must be specified for mode='radius'"):
            SpatialNeighborSearch(mode='radius')
    
    def test_init_invalid_mode_raises_error(self) -> None:
        """Test that invalid mode raises ValueError"""
        with pytest.raises(ValueError, match="mode must be 'radius', 'knn', or 'full', got invalid"):
            SpatialNeighborSearch(mode='invalid')
    
    def test_init_custom_max_num_neighbors(self) -> None:
        """Test initialization with custom max_num_neighbors"""
        search = SpatialNeighborSearch(mode='knn', k=10, max_num_neighbors=64)
        assert search.max_num_neighbors == 64
        
    def test_radius_example(self) -> None:
        """Test radius mode example initialization"""
        search = SpatialNeighborSearch(mode='radius', radius=1.0)
        
        coord1 = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        
        mask = search(coord1, return_full_mask=True)
        edges = search(coord1)
        assert mask.shape == (1, 2, 2)
        assert torch.sum(mask).item() == 2
        assert edges.shape[0] == 2
        assert edges.shape[1] == torch.sum(mask)
        
        coord2 = torch.tensor([[[10.0, 10.0], [2.0, 2.0]]])
        mask2 = search(coord1, coord2, return_full_mask=True)
        edges2 = search(coord1, coord2)
        assert mask2.shape == (1, 2, 2)
        assert torch.sum(mask2).item() == 0
        assert edges2.shape[0] == 2
        assert edges2.shape[1] == torch.sum(mask2)
        
        
    def test_knn_example(self) -> None:
        """Test knn mode example initialization"""
        search = SpatialNeighborSearch(mode='knn', k=32)
        
        coord1 = torch.randn((2, 100, 2))
        
        mask = search(coord1, return_full_mask=True)
        edges = search(coord1)
        assert mask.shape == (2, 100, 100)
        # Each point should have exactly k neighbors (excluding self)
        for b in range(2):
            for n in range(100):
                assert torch.sum(mask[b, :, n]).item() == 32
        assert edges.shape[0] == 2
        assert edges.shape[1] == torch.sum(mask)
                
        coord2 = torch.randn((2, 150, 2))
        mask2 = search(coord1, coord2, return_full_mask=True)
        edges2 = search(coord1, coord2)
        assert mask2.shape == (2, 100, 150)
        assert edges2.shape[0] == 2
        assert edges2.shape[1] == torch.sum(mask2)
        for b in range(2):
            for n in range(150):
                assert torch.sum(mask2[b, :, n]).item() == 32
                
    def test_full_example(self) -> None:
        """Test full example with both modes and return_full_mask option"""
        search_full = SpatialNeighborSearch(mode='full')
        
        coords_a = torch.randn((1, 50, 3))
        coords_b = torch.randn((1, 60, 3))
        
        mask_full = search_full(coords_a, coords_b, return_full_mask=True)
        edges_full = search_full(coords_a, coords_b)
        assert mask_full.shape == (1, 50, 60)
        assert torch.all(mask_full)
        assert edges_full.shape[1] == 50 * 60
        assert edges_full.shape[0] == 2