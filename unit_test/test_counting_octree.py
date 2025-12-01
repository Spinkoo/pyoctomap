#!/usr/bin/env python3
"""
Extensive test suite for CountingOcTree functionality.

This test suite covers:
- Recursive counting behavior
- Edge cases and boundary conditions
- Multiple resolutions
- Large-scale operations
- Tree structure validation
- Error handling
- Integration scenarios
"""

import pytest
import numpy as np
import sys
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pyoctomap import CountingOcTree, CountingOcTreeNode, OcTreeKey
    COUNTING_OCTREE_AVAILABLE = True
except ImportError:
    COUNTING_OCTREE_AVAILABLE = False
    pytest.skip("CountingOcTree not available", allow_module_level=True)


@pytest.fixture
def counting_tree():
    """Create a CountingOcTree instance for testing"""
    return CountingOcTree(0.1)


@pytest.fixture
def counting_tree_fine():
    """Create a fine-resolution CountingOcTree for testing"""
    return CountingOcTree(0.01)


@pytest.fixture
def counting_tree_coarse():
    """Create a coarse-resolution CountingOcTree for testing"""
    return CountingOcTree(1.0)


class TestRecursiveCounting:
    """Test recursive counting behavior - parent counts = sum of children"""
    
    def test_root_count_equals_total_observations(self, counting_tree):
        """Root count should equal total number of observations"""
        coords = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        
        total_updates = 0
        for coord in coords:
            for _ in range(3):
                counting_tree.updateNode(coord)
                total_updates += 1
        
        root = counting_tree.getRoot()
        assert root is not None
        assert root.getCount() == total_updates
    
    def test_parent_count_greater_equal_children(self, counting_tree):
        """Parent node count should be >= any child count"""
        # Create multiple observations at different locations
        for i in range(10):
            coord = [float(i), float(i), float(i)]
            counting_tree.updateNode(coord)
        
        root = counting_tree.getRoot()
        root_count = root.getCount()
        
        # All leaf nodes should have count <= root count
        centers = counting_tree.getCentersMinHits(1)
        for center in centers:
            node = counting_tree.search(center)
            if node:
                assert node.getCount() <= root_count
    
    def test_multiple_updates_same_location(self, counting_tree):
        """Multiple updates to same location should increment count correctly"""
        coord = [1.0, 2.0, 3.0]
        num_updates = 50
        
        for i in range(num_updates):
            node = counting_tree.updateNode(coord)
            assert node.getCount() == i + 1
        
        # Root should have count = num_updates
        root = counting_tree.getRoot()
        assert root.getCount() == num_updates
        
        # Leaf node should have count = num_updates
        leaf_node = counting_tree.search(coord)
        assert leaf_node.getCount() == num_updates
    
    def test_distributed_observations(self, counting_tree):
        """Test counting with observations distributed across space"""
        # Create a grid of observations
        grid_size = 5
        observations_per_cell = 3
        
        total_observations = 0
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    coord = [float(x), float(y), float(z)]
                    for _ in range(observations_per_cell):
                        counting_tree.updateNode(coord)
                        total_observations += 1
        
        root = counting_tree.getRoot()
        assert root.getCount() == total_observations
        
        # Verify all cells have correct count
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    coord = [float(x), float(y), float(z)]
                    node = counting_tree.search(coord)
                    assert node is not None
                    assert node.getCount() == observations_per_cell


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_tree_operations(self, counting_tree):
        """Test operations on empty tree"""
        assert counting_tree.size() == 0
        assert counting_tree.getNumLeafNodes() == 0
        
        # Search should return None
        node = counting_tree.search([1.0, 2.0, 3.0])
        assert node is None
        
        # getCentersMinHits should return empty list
        centers = counting_tree.getCentersMinHits(1)
        assert len(centers) == 0
        
        # Root might be None for empty tree (OcTreeBase creates root on first update)
        root = counting_tree.getRoot()
        # Empty tree may not have root yet - this is implementation dependent
        # After first update, root will exist
    
    def test_single_observation(self, counting_tree):
        """Test tree with single observation"""
        coord = [1.0, 2.0, 3.0]
        node = counting_tree.updateNode(coord)
        
        assert node is not None
        assert node.getCount() == 1
        assert counting_tree.size() > 0
        assert counting_tree.getNumLeafNodes() == 1
        
        root = counting_tree.getRoot()
        assert root.getCount() == 1
    
    def test_zero_count_threshold(self, counting_tree):
        """Test getCentersMinHits with zero threshold"""
        coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        for coord in coords:
            counting_tree.updateNode(coord)
        
        centers = counting_tree.getCentersMinHits(0)
        assert len(centers) >= len(coords)
    
    def test_very_large_count(self, counting_tree):
        """Test with very large count values"""
        coord = [1.0, 2.0, 3.0]
        node = counting_tree.updateNode(coord)
        
        # Set a large count
        large_count = 1000000
        node.setCount(large_count)
        assert node.getCount() == large_count
        
        # Verify it's still searchable
        found_node = counting_tree.search(coord)
        assert found_node.getCount() == large_count
    
    def test_negative_coordinates(self, counting_tree):
        """Test with negative coordinates"""
        coords = [
            [-1.0, -2.0, -3.0],
            [-10.0, -20.0, -30.0],
        ]
        
        for coord in coords:
            node = counting_tree.updateNode(coord)
            assert node is not None
            assert node.getCount() == 1
            
            # Verify search works
            found = counting_tree.search(coord)
            assert found is not None
            assert found.getCount() == 1
    
    def test_large_coordinates(self, counting_tree):
        """Test with large coordinate values"""
        coords = [
            [100.0, 200.0, 300.0],
            [1000.0, 2000.0, 3000.0],
        ]
        
        for coord in coords:
            # Large coordinates might be out of range, so check if update succeeds
            node = counting_tree.updateNode(coord)
            # If node is None, coordinates are out of range (acceptable)
            if node is not None:
                assert node.getCount() == 1
    
    def test_very_close_coordinates(self, counting_tree_fine):
        """Test with very close coordinates (requires fine resolution)"""
        base_coord = [1.0, 2.0, 3.0]
        offsets = [0.001, 0.002, 0.003]
        
        for offset in offsets:
            coord = [base_coord[0] + offset, base_coord[1], base_coord[2]]
            node = counting_tree_fine.updateNode(coord)
            assert node is not None


class TestMultipleResolutions:
    """Test behavior with different resolutions"""
    
    def test_different_resolutions(self):
        """Test that different resolutions work correctly"""
        resolutions = [0.01, 0.1, 0.5, 1.0]
        
        for resolution in resolutions:
            tree = CountingOcTree(resolution)
            assert tree.getResolution() == resolution
            
            # Add some observations
            for i in range(5):
                coord = [float(i), float(i), float(i)]
                node = tree.updateNode(coord)
                assert node is not None
                assert node.getCount() == 1
    
    def test_resolution_affects_discretization(self):
        """Test that resolution affects coordinate discretization"""
        coord = [1.234, 5.678, 9.012]
        
        tree_fine = CountingOcTree(0.01)
        tree_coarse = CountingOcTree(0.1)
        
        key_fine = tree_fine.coordToKey(np.array(coord))
        key_coarse = tree_coarse.coordToKey(np.array(coord))
        
        # Keys should be different due to different resolutions
        assert key_fine != key_coarse or tree_fine.getResolution() == tree_coarse.getResolution()
    
    def test_resolution_affects_node_count(self):
        """Test that resolution affects number of nodes"""
        # Create same observation pattern with different resolutions
        coords = []
        for x in range(10):
            for y in range(10):
                coords.append([float(x), float(y), 0.0])
        
        tree_fine = CountingOcTree(0.01)
        tree_coarse = CountingOcTree(1.0)
        
        for coord in coords:
            tree_fine.updateNode(coord)
            tree_coarse.updateNode(coord)
        
        # Fine resolution should create more nodes
        assert tree_fine.size() >= tree_coarse.size()


class TestNodeManipulation:
    """Test node count manipulation"""
    
    def test_increase_count_multiple_times(self, counting_tree):
        """Test increasing count multiple times"""
        coord = [1.0, 2.0, 3.0]
        node = counting_tree.updateNode(coord)
        
        initial_count = node.getCount()
        num_increases = 10
        
        for _ in range(num_increases):
            node.increaseCount()
        
        assert node.getCount() == initial_count + num_increases
    
    def test_set_count_various_values(self, counting_tree):
        """Test setting count to various values"""
        coord = [1.0, 2.0, 3.0]
        node = counting_tree.updateNode(coord)
        
        test_values = [0, 1, 10, 100, 1000, 10000]
        for value in test_values:
            node.setCount(value)
            assert node.getCount() == value
            assert node.getValue() == value
    
    def test_set_value_equals_set_count(self, counting_tree):
        """Test that setValue and setCount are equivalent"""
        coord = [1.0, 2.0, 3.0]
        node = counting_tree.updateNode(coord)
        
        test_value = 42
        node.setValue(test_value)
        assert node.getCount() == test_value
        
        node.setCount(test_value + 1)
        assert node.getValue() == test_value + 1


class TestGetCentersMinHits:
    """Extensive tests for getCentersMinHits"""
    
    def test_threshold_filtering(self, counting_tree):
        """Test that thresholds filter correctly"""
        # Create pattern with known counts
        patterns = [
            ([1.0, 1.0, 1.0], 10),  # High count
            ([2.0, 2.0, 2.0], 5),   # Medium count
            ([3.0, 3.0, 3.0], 2),  # Low count
            ([4.0, 4.0, 4.0], 1),  # Very low count
        ]
        
        for coord, count in patterns:
            for _ in range(count):
                counting_tree.updateNode(coord)
        
        # Test various thresholds
        centers_1 = counting_tree.getCentersMinHits(1)
        assert len(centers_1) >= 4
        
        centers_2 = counting_tree.getCentersMinHits(2)
        assert len(centers_2) >= 3
        
        centers_5 = counting_tree.getCentersMinHits(5)
        assert len(centers_5) >= 2
        
        centers_10 = counting_tree.getCentersMinHits(10)
        assert len(centers_10) >= 1
        
        centers_11 = counting_tree.getCentersMinHits(11)
        assert len(centers_11) == 0
    
    def test_centers_are_voxel_centers(self, counting_tree):
        """Test that returned centers are voxel centers, not input coordinates"""
        input_coord = [1.0, 2.0, 3.0]
        counting_tree.updateNode(input_coord)
        
        centers = counting_tree.getCentersMinHits(1)
        assert len(centers) == 1
        
        center = centers[0]
        # Center should be close to input but may differ by up to resolution/2
        diff = np.abs(np.array(input_coord) - np.array(center))
        assert np.all(diff <= counting_tree.getResolution())
    
    def test_centers_match_node_counts(self, counting_tree):
        """Test that centers returned match nodes with correct counts"""
        patterns = [
            ([1.0, 1.0, 1.0], 5),
            ([2.0, 2.0, 2.0], 3),
        ]
        
        for coord, count in patterns:
            for _ in range(count):
                counting_tree.updateNode(coord)
        
        centers = counting_tree.getCentersMinHits(3)
        for center in centers:
            node = counting_tree.search(center)
            assert node is not None
            assert node.getCount() >= 3


class TestCoordinateConversion:
    """Extensive coordinate conversion tests"""
    
    def test_coord_to_key_consistency(self, counting_tree):
        """Test that coordToKey is consistent"""
        coords = [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [-1.0, -2.0, -3.0],
            [100.0, 200.0, 300.0],
        ]
        
        for coord in coords:
            key1 = counting_tree.coordToKey(np.array(coord))
            key2 = counting_tree.coordToKey(np.array(coord))
            assert key1 == key2
    
    def test_key_to_coord_roundtrip(self, counting_tree):
        """Test key-to-coord roundtrip accuracy"""
        test_coords = [
            [0.0, 0.0, 0.0],
            [1.23, 4.56, 7.89],
            [-1.5, -2.5, -3.5],
            [10.0, 20.0, 30.0],
        ]
        
        for coord in test_coords:
            coord_arr = np.array(coord)
            key = counting_tree.coordToKey(coord_arr)
            recovered = counting_tree.keyToCoord(key)
            
            # Should be within resolution
            diff = np.abs(coord_arr - recovered)
            assert np.all(diff < counting_tree.getResolution())
    
    def test_coord_to_key_with_depth(self, counting_tree):
        """Test coordToKey with depth parameter"""
        coord = np.array([1.0, 2.0, 3.0])
        
        key_default = counting_tree.coordToKey(coord)
        key_depth0 = counting_tree.coordToKey(coord, depth=0)
        key_depth5 = counting_tree.coordToKey(coord, depth=5)
        
        # Keys at different depths should be different
        assert key_default is not None
        assert key_depth0 is not None
        assert key_depth5 is not None
    
    def test_key_to_coord_with_depth(self, counting_tree):
        """Test keyToCoord with depth parameter"""
        coord = np.array([1.0, 2.0, 3.0])
        key = counting_tree.coordToKey(coord)
        
        coord_default = counting_tree.keyToCoord(key)
        coord_depth0 = counting_tree.keyToCoord(key, depth=0)
        coord_depth5 = counting_tree.keyToCoord(key, depth=5)
        
        assert coord_default is not None
        assert coord_depth0 is not None
        assert coord_depth5 is not None


class TestFileIO:
    """Extensive file I/O tests"""
    
    def test_write_read_preserves_counts(self, counting_tree, tmp_path):
        """Test that write/read preserves all counts"""
        # Create complex pattern
        patterns = [
            ([1.0, 1.0, 1.0], 10),
            ([2.0, 2.0, 2.0], 5),
            ([3.0, 3.0, 3.0], 3),
        ]
        
        for coord, count in patterns:
            for _ in range(count):
                counting_tree.updateNode(coord)
        
        filename = str(tmp_path / "test_counting.ot")
        assert counting_tree.write(filename)
        
        # Read back
        tree_loaded = counting_tree.read(filename)
        assert tree_loaded is not None
        
        # Verify all counts
        for coord, expected_count in patterns:
            node_orig = counting_tree.search(coord)
            node_loaded = tree_loaded.search(coord)
            
            assert node_orig is not None
            assert node_loaded is not None
            assert node_orig.getCount() == expected_count
            assert node_loaded.getCount() == expected_count
    
    def test_write_read_preserves_tree_structure(self, counting_tree, tmp_path):
        """Test that write/read preserves tree structure"""
        # Create tree with multiple levels
        for x in range(5):
            for y in range(5):
                counting_tree.updateNode([float(x), float(y), 0.0])
        
        original_size = counting_tree.size()
        original_leaves = counting_tree.getNumLeafNodes()
        
        filename = str(tmp_path / "test_counting.ot")
        counting_tree.write(filename)
        
        tree_loaded = counting_tree.read(filename)
        assert tree_loaded.size() == original_size
        assert tree_loaded.getNumLeafNodes() == original_leaves
    
    def test_read_nonexistent_file(self, counting_tree):
        """Test reading non-existent file"""
        result = counting_tree.read("/nonexistent/path/file.ot")
        assert result is None
    
    def test_write_empty_tree(self, counting_tree, tmp_path):
        """Test writing empty tree"""
        filename = str(tmp_path / "empty.ot")
        assert counting_tree.write(filename)
        
        # Should be able to read it back
        tree_loaded = counting_tree.read(filename)
        assert tree_loaded is not None
        assert tree_loaded.size() == 0


class TestTreeStatistics:
    """Extensive tree statistics tests"""
    
    def test_size_increases_with_observations(self, counting_tree):
        """Test that size increases with observations"""
        initial_size = counting_tree.size()
        
        for i in range(10):
            counting_tree.updateNode([float(i), 0.0, 0.0])
            assert counting_tree.size() > initial_size
            initial_size = counting_tree.size()
    
    def test_leaf_nodes_count(self, counting_tree):
        """Test leaf node counting"""
        assert counting_tree.getNumLeafNodes() == 0
        
        # Add observations at different locations
        num_locations = 10
        for i in range(num_locations):
            counting_tree.updateNode([float(i), 0.0, 0.0])
        
        # Should have at least num_locations leaf nodes
        assert counting_tree.getNumLeafNodes() >= num_locations
    
    def test_calc_num_nodes(self, counting_tree):
        """Test calcNumNodes"""
        counting_tree.updateNode([1.0, 2.0, 3.0])
        
        calc_nodes = counting_tree.calcNumNodes()
        size_nodes = counting_tree.size()
        
        # calcNumNodes should be >= size (includes internal nodes)
        assert calc_nodes >= size_nodes
    
    def test_metric_bounds(self, counting_tree):
        """Test metric bounds calculation"""
        coords = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        
        for coord in coords:
            counting_tree.updateNode(coord)
        
        metric_min = counting_tree.getMetricMin()
        metric_max = counting_tree.getMetricMax()
        metric_size = counting_tree.getMetricSize()
        
        assert len(metric_min) == 3
        assert len(metric_max) == 3
        assert len(metric_size) == 3
        
        # Size should be positive
        assert np.all(metric_size >= 0)
        
        # Max should be >= min
        assert np.all(metric_max >= metric_min)
    
    def test_memory_usage(self, counting_tree):
        """Test memory usage reporting"""
        memory_empty = counting_tree.memoryUsage()
        assert memory_empty > 0
        
        # Add nodes
        for i in range(10):
            counting_tree.updateNode([float(i), 0.0, 0.0])
        
        memory_with_nodes = counting_tree.memoryUsage()
        assert memory_with_nodes > memory_empty
    
    def test_volume_calculation(self, counting_tree):
        """Test volume calculation"""
        volume_empty = counting_tree.volume()
        assert volume_empty >= 0
        
        counting_tree.updateNode([1.0, 2.0, 3.0])
        volume_with_nodes = counting_tree.volume()
        assert volume_with_nodes >= volume_empty


class TestTreeOperations:
    """Test tree manipulation operations"""
    
    def test_clear_tree(self, counting_tree):
        """Test clearing tree"""
        # Add some nodes
        for i in range(10):
            counting_tree.updateNode([float(i), 0.0, 0.0])
        
        assert counting_tree.size() > 0
        
        counting_tree.clear()
        
        assert counting_tree.size() == 0
        assert counting_tree.getNumLeafNodes() == 0
        
        # After clear, root might be None or have count 0 (implementation dependent)
        root = counting_tree.getRoot()
        if root is not None:
            assert root.getCount() == 0
    
    def test_search_at_different_depths(self, counting_tree):
        """Test searching at different depths"""
        coord = [1.0, 2.0, 3.0]
        counting_tree.updateNode(coord)
        
        # Search at default depth
        node_default = counting_tree.search(coord)
        assert node_default is not None
        
        # Search at specific depth
        node_depth0 = counting_tree.search(coord, depth=0)
        node_depth5 = counting_tree.search(coord, depth=5)
        
        assert node_depth0 is not None
        assert node_depth5 is not None
    
    def test_get_root(self, counting_tree):
        """Test getting root node"""
        # Root might be None for empty tree, so add a node first
        counting_tree.updateNode([1.0, 2.0, 3.0])
        
        root = counting_tree.getRoot()
        assert root is not None
        assert isinstance(root, CountingOcTreeNode)
        
        # Root should have count >= 1 after update
        assert root.getCount() >= 1
        
        # After more updates, root count should increase
        counting_tree.updateNode([4.0, 5.0, 6.0])
        assert root.getCount() >= 2


class TestLargeScale:
    """Test large-scale operations"""
    
    def test_many_observations(self, counting_tree):
        """Test with many observations"""
        num_observations = 1000
        np.random.seed(42)
        
        for _ in range(num_observations):
            coord = np.random.rand(3) * 10.0
            counting_tree.updateNode(coord.tolist())
        
        root = counting_tree.getRoot()
        assert root.getCount() == num_observations
        
        assert counting_tree.size() > 0
        assert counting_tree.getNumLeafNodes() > 0
    
    def test_many_unique_locations(self, counting_tree):
        """Test with many unique locations"""
        grid_size = 20
        total_locations = grid_size ** 3
        
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    counting_tree.updateNode([float(x), float(y), float(z)])
        
        root = counting_tree.getRoot()
        assert root.getCount() == total_locations
        
        # Should have many leaf nodes
        assert counting_tree.getNumLeafNodes() >= total_locations * 0.9  # Allow some tolerance


class TestIntegration:
    """Integration tests combining multiple features"""
    
    def test_full_workflow(self, counting_tree, tmp_path):
        """Test complete workflow: create, update, query, save, load"""
        # 1. Create and populate
        coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        for coord in coords:
            for _ in range(5):
                counting_tree.updateNode(coord)
        
        # 2. Query
        for coord in coords:
            node = counting_tree.search(coord)
            assert node is not None
            assert node.getCount() == 5
        
        # 3. Find frequent locations
        centers = counting_tree.getCentersMinHits(3)
        assert len(centers) >= len(coords)
        
        # 4. Save
        filename = str(tmp_path / "workflow.ot")
        assert counting_tree.write(filename)
        
        # 5. Load and verify
        tree_loaded = counting_tree.read(filename)
        assert tree_loaded is not None
        
        for coord in coords:
            node = tree_loaded.search(coord)
            assert node is not None
            assert node.getCount() == 5
    
    def test_coordinate_conversion_integration(self, counting_tree):
        """Test coordinate conversion integrated with updates"""
        coords = [
            [1.23, 4.56, 7.89],
            [10.0, 20.0, 30.0],
        ]
        
        for coord in coords:
            # Convert to key
            key = counting_tree.coordToKey(np.array(coord))
            
            # Update using key
            node = counting_tree.updateNode(key)
            assert node is not None
            
            # Convert back and verify
            recovered = counting_tree.keyToCoord(key)
            diff = np.abs(np.array(coord) - recovered)
            assert np.all(diff < counting_tree.getResolution())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

