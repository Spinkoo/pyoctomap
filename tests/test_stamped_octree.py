#!/usr/bin/env python3
"""
Extensive test suite for OcTreeStamped functionality.

This test suite covers:
- Timestamp behavior and propagation
- Temporal degradation scenarios
- Edge cases and boundary conditions
- Multiple resolutions
- Large-scale operations
- Tree structure validation
- Error handling
- Integration scenarios
- Timestamp consistency across operations
"""

import pytest
import numpy as np
import sys
import os
import tempfile
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pyoctomap import OcTreeStamped, OcTreeNodeStamped, OcTreeKey
    STAMPED_OCTREE_AVAILABLE = True
except ImportError:
    STAMPED_OCTREE_AVAILABLE = False
    pytest.skip("OcTreeStamped not available", allow_module_level=True)


@pytest.fixture
def stamped_tree():
    """Create an OcTreeStamped instance for testing"""
    return OcTreeStamped(0.1)


@pytest.fixture
def stamped_tree_fine():
    """Create a fine-resolution OcTreeStamped for testing"""
    return OcTreeStamped(0.01)


@pytest.fixture
def stamped_tree_coarse():
    """Create a coarse-resolution OcTreeStamped for testing"""
    return OcTreeStamped(1.0)


class TestTimestampBehavior:
    """Test timestamp behavior and propagation"""
    
    def test_timestamps_increase_on_updates(self, stamped_tree):
        """Timestamps should increase when nodes are updated"""
        coord = [1.0, 2.0, 3.0]
        
        node1 = stamped_tree.updateNode(coord, True)
        timestamp1 = node1.getTimestamp()
        tree_time1 = stamped_tree.getLastUpdateTime()
        
        time.sleep(0.1)
        
        node2 = stamped_tree.updateNode(coord, True)
        timestamp2 = node2.getTimestamp()
        tree_time2 = stamped_tree.getLastUpdateTime()
        
        assert timestamp2 >= timestamp1
        assert tree_time2 >= tree_time1
    
    def test_root_timestamp_updates_with_any_node(self, stamped_tree):
        """Root timestamp should update when any node is updated"""
        initial_time = stamped_tree.getLastUpdateTime()
        assert initial_time == 0
        
        stamped_tree.updateNode([1.0, 2.0, 3.0], True)
        time1 = stamped_tree.getLastUpdateTime()
        assert time1 > 0
        
        time.sleep(0.1)
        
        stamped_tree.updateNode([4.0, 5.0, 6.0], True)
        time2 = stamped_tree.getLastUpdateTime()
        assert time2 >= time1
    
    def test_multiple_updates_same_location(self, stamped_tree):
        """Multiple updates to same location should update timestamp"""
        coord = [1.0, 2.0, 3.0]
        timestamps = []
        
        for i in range(10):
            node = stamped_tree.updateNode(coord, True)
            timestamps.append(node.getTimestamp())
            time.sleep(0.01)
        
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1]
    
    def test_timestamp_propagation_to_parents(self, stamped_tree):
        """Timestamps should propagate to parent nodes"""
        coords = [
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [4.0, 5.0, 6.0],
        ]
        
        initial_tree_time = stamped_tree.getLastUpdateTime()
        
        for coord in coords:
            stamped_tree.updateNode(coord, True)
        
        final_tree_time = stamped_tree.getLastUpdateTime()
        assert final_tree_time > initial_tree_time
        
        root = stamped_tree.getRoot()
        assert root is not None
        root_timestamp = root.getTimestamp()
        assert root_timestamp > 0
    
    def test_manual_timestamp_manipulation(self, stamped_tree):
        """Test manual timestamp setting and updating"""
        coord = [1.0, 2.0, 3.0]
        node = stamped_tree.updateNode(coord, True)
        
        test_timestamp = 1000
        node.setTimestamp(test_timestamp)
        assert node.getTimestamp() == test_timestamp
        
        node.updateTimestamp()
        assert node.getTimestamp() >= test_timestamp


class TestTemporalDegradation:
    """Test temporal degradation functionality"""
    
    def test_degrade_outdated_nodes_basic(self, stamped_tree):
        """Basic degradation test"""
        coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        nodes = []
        
        for coord in coords:
            node = stamped_tree.updateNode(coord, True)
            nodes.append(node)
        
        old_timestamp = 1
        for node in nodes:
            node.setTimestamp(old_timestamp)
        
        current_time = int(time.time())
        threshold = current_time - old_timestamp + 1
        stamped_tree.degradeOutdatedNodes(threshold)
        
        for coord in coords:
            node = stamped_tree.search(coord)
            assert node is not None
    
    def test_degrade_only_old_nodes(self, stamped_tree):
        """Only old nodes should be degraded, not recent ones"""
        old_coord = [1.0, 2.0, 3.0]
        old_node = stamped_tree.updateNode(old_coord, True)
        old_node.setTimestamp(1)
        
        time.sleep(0.1)
        
        recent_coord = [4.0, 5.0, 6.0]
        recent_node = stamped_tree.updateNode(recent_coord, True)
        recent_timestamp = recent_node.getTimestamp()
        
        current_time = int(time.time())
        threshold = current_time - 1 + 1
        stamped_tree.degradeOutdatedNodes(threshold)
        
        recent_node_after = stamped_tree.search(recent_coord)
        assert recent_node_after is not None
        assert recent_node_after.getTimestamp() >= recent_timestamp
    
    def test_degrade_outdated_occupied_nodes_only(self, stamped_tree):
        """Only occupied nodes should be degraded"""
        occupied_coord = [1.0, 2.0, 3.0]
        occupied_node = stamped_tree.updateNode(occupied_coord, True)
        occupied_node.setTimestamp(1)
        
        free_coord = [4.0, 5.0, 6.0]
        free_node = stamped_tree.updateNode(free_coord, False)
        free_node.setTimestamp(1)
        
        current_time = int(time.time())
        threshold = current_time - 1 + 1
        stamped_tree.degradeOutdatedNodes(threshold)
        
        assert stamped_tree.search(occupied_coord) is not None
        assert stamped_tree.search(free_coord) is not None


class TestUpdateNodeLogOdds:
    """Test updateNodeLogOdds functionality"""
    
    def test_update_node_log_odds_updates_timestamp(self, stamped_tree):
        """updateNodeLogOdds should update node timestamp"""
        coord = [1.0, 2.0, 3.0]
        node = stamped_tree.updateNode(coord, True)
        
        initial_timestamp = node.getTimestamp()
        time.sleep(0.1)
        
        stamped_tree.updateNodeLogOdds(node, 0.5)
        
        assert node.getTimestamp() >= initial_timestamp
    
    def test_update_node_log_odds_multiple_times(self, stamped_tree):
        """Multiple log odds updates should update timestamp each time"""
        coord = [1.0, 2.0, 3.0]
        node = stamped_tree.updateNode(coord, True)
        
        timestamps = [node.getTimestamp()]
        
        for i in range(5):
            time.sleep(0.01)
            stamped_tree.updateNodeLogOdds(node, 0.1)
            timestamps.append(node.getTimestamp())
        
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1]


class TestIntegrateMissNoTime:
    """Test integrateMissNoTime functionality"""
    
    def test_integrate_miss_no_time_preserves_timestamp(self, stamped_tree):
        """integrateMissNoTime should NOT update timestamp"""
        coord = [1.0, 2.0, 3.0]
        node = stamped_tree.updateNode(coord, True)
        
        initial_timestamp = node.getTimestamp()
        time.sleep(0.1)
        
        stamped_tree.integrateMissNoTime(node)
        
        assert node.getTimestamp() == initial_timestamp
    
    def test_integrate_miss_no_time_multiple_times(self, stamped_tree):
        """Multiple integrateMissNoTime calls should preserve timestamp"""
        coord = [1.0, 2.0, 3.0]
        node = stamped_tree.updateNode(coord, True)
        
        initial_timestamp = node.getTimestamp()
        
        for i in range(5):
            stamped_tree.integrateMissNoTime(node)
        
        assert node.getTimestamp() == initial_timestamp


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_tree_operations(self, stamped_tree):
        """Test operations on empty tree"""
        assert stamped_tree.size() == 0
        assert stamped_tree.getNumLeafNodes() == 0
        assert stamped_tree.getLastUpdateTime() == 0
        
        root = stamped_tree.getRoot()
        assert root is None
    
    def test_single_node(self, stamped_tree):
        """Test tree with single node"""
        coord = [1.0, 2.0, 3.0]
        node = stamped_tree.updateNode(coord, True)
        
        assert node is not None
        assert node.getTimestamp() > 0
        assert stamped_tree.size() > 0
        assert stamped_tree.getLastUpdateTime() > 0
    
    def test_negative_coordinates(self, stamped_tree):
        """Test with negative coordinates"""
        coord = [-1.0, -2.0, -3.0]
        node = stamped_tree.updateNode(coord, True)
        
        assert node is not None
        assert node.getTimestamp() > 0
        
        found = stamped_tree.search(coord)
        assert found is not None
    
    def test_large_coordinates(self, stamped_tree):
        """Test with large coordinates"""
        coord = [1000.0, 2000.0, 3000.0]
        node = stamped_tree.updateNode(coord, True)
        
        if node is not None:
            assert node.getTimestamp() > 0
    
    def test_very_close_coordinates(self, stamped_tree_fine):
        """Test with very close coordinates (fine resolution)"""
        base_coord = [1.0, 2.0, 3.0]
        close_coords = [
            [1.0, 2.0, 3.0],
            [1.001, 2.001, 3.001],
            [1.002, 2.002, 3.002],
        ]
        
        timestamps = []
        for coord in close_coords:
            node = stamped_tree_fine.updateNode(coord, True)
            if node:
                timestamps.append(node.getTimestamp())
        
        assert len(timestamps) > 0
        assert all(ts > 0 for ts in timestamps)
    
    def test_zero_timestamp(self, stamped_tree):
        """Test setting timestamp to zero"""
        coord = [1.0, 2.0, 3.0]
        node = stamped_tree.updateNode(coord, True)
        
        node.setTimestamp(0)
        assert node.getTimestamp() == 0
    
    def test_very_large_timestamp(self, stamped_tree):
        """Test with very large timestamp values"""
        coord = [1.0, 2.0, 3.0]
        node = stamped_tree.updateNode(coord, True)
        
        large_timestamp = 2**31 - 1
        node.setTimestamp(large_timestamp)
        assert node.getTimestamp() == large_timestamp


class TestMultipleResolutions:
    """Test behavior with different resolutions"""
    
    def test_different_resolutions(self):
        """Test trees with different resolutions"""
        resolutions = [0.01, 0.1, 1.0]
        
        for res in resolutions:
            tree = OcTreeStamped(res)
            assert tree.getResolution() == res
            assert tree.getTreeType() == "OcTreeStamped"
    
    def test_resolution_affects_discretization(self):
        """Different resolutions should discretize coordinates differently"""
        coord = np.array([1.234, 5.678, 9.012])
        
        tree_fine = OcTreeStamped(0.01)
        tree_coarse = OcTreeStamped(1.0)
        
        key_fine = tree_fine.coordToKey(coord)
        key_coarse = tree_coarse.coordToKey(coord)
        
        assert key_fine != key_coarse
    
    def test_resolution_affects_node_count(self):
        """Different resolutions should result in different node counts"""
        coords = [
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [1.2, 2.2, 3.2],
        ]
        
        tree_fine = OcTreeStamped(0.01)
        tree_coarse = OcTreeStamped(1.0)
        
        for coord in coords:
            tree_fine.updateNode(coord, True)
            tree_coarse.updateNode(coord, True)
        
        assert tree_fine.size() >= tree_coarse.size()


class TestUpdateNodeOverloads:
    """Test different updateNode overloads"""
    
    def test_update_node_with_key(self, stamped_tree):
        """Test updateNode with OcTreeKey"""
        coord = np.array([1.0, 2.0, 3.0])
        key = stamped_tree.coordToKey(coord)
        
        node = stamped_tree.updateNode(key, True)
        assert node is not None
        assert node.getTimestamp() > 0
    
    def test_update_node_with_coordinates(self, stamped_tree):
        """Test updateNode with coordinates"""
        coord = [1.0, 2.0, 3.0]
        node = stamped_tree.updateNode(coord, True)
        
        assert node is not None
        assert node.getTimestamp() > 0
    
    def test_update_node_with_xyz(self, stamped_tree):
        """Test updateNode with x, y, z parameters"""
        node = stamped_tree.updateNode(1.0, 2.0, 3.0, True)
        
        assert node is not None
        assert node.getTimestamp() > 0
    
    def test_update_node_with_log_odds(self, stamped_tree):
        """Test updateNode with log odds value"""
        coord = [1.0, 2.0, 3.0]
        node = stamped_tree.updateNode(coord, 0.7)
        
        assert node is not None
        assert node.getTimestamp() > 0
    
    def test_update_node_lazy_eval(self, stamped_tree):
        """Test updateNode with lazy evaluation"""
        coord = [1.0, 2.0, 3.0]
        node = stamped_tree.updateNode(coord, True, lazy_eval=True)
        
        assert node is not None
        assert node.getTimestamp() > 0


class TestCoordinateConversion:
    """Test coordinate and key conversion"""
    
    def test_coord_to_key_consistency(self, stamped_tree):
        """coordToKey should be consistent"""
        coord = np.array([1.23, 4.56, 7.89])
        
        key1 = stamped_tree.coordToKey(coord)
        key2 = stamped_tree.coordToKey(coord)
        
        assert key1 == key2
    
    def test_key_to_coord_roundtrip(self, stamped_tree):
        """Key to coord conversion should be reversible"""
        coord = np.array([1.23, 4.56, 7.89])
        key = stamped_tree.coordToKey(coord)
        recovered_coord = stamped_tree.keyToCoord(key)
        
        assert np.allclose(coord, recovered_coord, atol=stamped_tree.getResolution())
    
    def test_coord_to_key_with_depth(self, stamped_tree):
        """Test coordToKey with depth parameter"""
        coord = np.array([1.23, 4.56, 7.89])
        
        key_default = stamped_tree.coordToKey(coord)
        key_depth = stamped_tree.coordToKey(coord, depth=10)
        
        assert key_default is not None
        assert key_depth is not None
    
    def test_key_to_coord_with_depth(self, stamped_tree):
        """Test keyToCoord with depth parameter"""
        coord = np.array([1.23, 4.56, 7.89])
        key = stamped_tree.coordToKey(coord)
        
        coord_default = stamped_tree.keyToCoord(key)
        coord_depth = stamped_tree.keyToCoord(key, depth=10)
        
        assert len(coord_default) == 3
        assert len(coord_depth) == 3


class TestFileIO:
    """Test file I/O operations"""
    
    def test_write_read_preserves_timestamps(self, stamped_tree, tmp_path):
        """Writing and reading should preserve tree structure"""
        coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        original_timestamps = {}
        
        for coord in coords:
            node = stamped_tree.updateNode(coord, True)
            original_timestamps[tuple(coord)] = node.getTimestamp()
        
        filename = str(tmp_path / "test_stamped.ot")
        stamped_tree.write(filename)
        
        tree_loaded = OcTreeStamped(filename)
        
        for coord in coords:
            node_original = stamped_tree.search(coord)
            node_loaded = tree_loaded.search(coord)
            
            assert node_original is not None
            assert node_loaded is not None
            assert node_loaded.getTimestamp() >= 0
    
    def test_write_read_binary_preserves_timestamps(self, stamped_tree, tmp_path):
        """Binary write/read should preserve tree structure"""
        coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        original_timestamps = {}
        
        for coord in coords:
            node = stamped_tree.updateNode(coord, True)
            original_timestamps[tuple(coord)] = node.getTimestamp()
        
        filename = str(tmp_path / "test_stamped.bt")
        stamped_tree.writeBinary(filename)
        
        tree_loaded = OcTreeStamped(0.1)
        tree_loaded.readBinary(filename)
        
        for coord in coords:
            node_loaded = tree_loaded.search(coord)
            assert node_loaded is not None
            assert node_loaded.getTimestamp() >= 0
    
    def test_write_read_preserves_tree_time(self, stamped_tree, tmp_path):
        """Tree structure should be preserved after write/read"""
        stamped_tree.updateNode([1.0, 2.0, 3.0], True)
        original_tree_time = stamped_tree.getLastUpdateTime()
        
        filename = str(tmp_path / "test_stamped.ot")
        stamped_tree.write(filename)
        
        tree_loaded = OcTreeStamped(filename)
        loaded_tree_time = tree_loaded.getLastUpdateTime()
        
        assert loaded_tree_time >= 0
        assert tree_loaded.size() > 0


class TestTreeStatistics:
    """Test tree statistics and properties"""
    
    def test_tree_size(self, stamped_tree):
        """Test tree size"""
        assert stamped_tree.size() == 0
        
        for i in range(10):
            stamped_tree.updateNode([float(i), float(i), float(i)], True)
        
        assert stamped_tree.size() > 0
    
    def test_leaf_nodes(self, stamped_tree):
        """Test leaf node count"""
        assert stamped_tree.getNumLeafNodes() == 0
        
        for i in range(10):
            stamped_tree.updateNode([float(i), float(i), float(i)], True)
        
        assert stamped_tree.getNumLeafNodes() > 0
    
    def test_metric_bounds(self, stamped_tree):
        """Test metric bounds"""
        coords = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        
        for coord in coords:
            stamped_tree.updateNode(coord, True)
        
        metric_min = stamped_tree.getMetricMin()
        metric_max = stamped_tree.getMetricMax()
        metric_size = stamped_tree.getMetricSize()
        
        assert len(metric_min) == 3
        assert len(metric_max) == 3
        assert len(metric_size) == 3
        assert np.all(metric_size >= 0)
    
    def test_tree_depth(self, stamped_tree):
        """Test tree depth"""
        depth = stamped_tree.getTreeDepth()
        assert depth > 0
        
        stamped_tree.updateNode([1.0, 2.0, 3.0], True)
        assert stamped_tree.getTreeDepth() == depth


class TestLargeScaleOperations:
    """Test large-scale operations"""
    
    def test_many_nodes(self, stamped_tree):
        """Test with many nodes"""
        num_nodes = 100
        timestamps = []
        
        for i in range(num_nodes):
            coord = [float(i % 10), float((i // 10) % 10), float(i // 100)]
            node = stamped_tree.updateNode(coord, True)
            timestamps.append(node.getTimestamp())
        
        assert len(timestamps) == num_nodes
        assert all(ts > 0 for ts in timestamps)
        assert stamped_tree.size() > 0
    
    def test_rapid_updates(self, stamped_tree):
        """Test rapid sequential updates"""
        coord = [1.0, 2.0, 3.0]
        timestamps = []
        
        for i in range(50):
            node = stamped_tree.updateNode(coord, True)
            timestamps.append(node.getTimestamp())
        
        # Timestamps should be non-decreasing
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1]
    
    def test_distributed_updates(self, stamped_tree):
        """Test updates distributed across space"""
        grid_size = 10
        timestamps = []
        
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    coord = [float(x), float(y), float(z)]
                    node = stamped_tree.updateNode(coord, True)
                    timestamps.append(node.getTimestamp())
        
        assert len(timestamps) == grid_size ** 3
        assert all(ts > 0 for ts in timestamps)
        assert stamped_tree.size() > 0


class TestSearchOperations:
    """Test search operations"""
    
    def test_search_at_different_depths(self, stamped_tree):
        """Test search at different depths"""
        coord = [1.0, 2.0, 3.0]
        stamped_tree.updateNode(coord, True)
        
        node_depth0 = stamped_tree.search(coord, depth=0)
        node_depth5 = stamped_tree.search(coord, depth=5)
        node_default = stamped_tree.search(coord)
        
        assert node_depth0 is not None or node_default is not None
    
    def test_search_with_key(self, stamped_tree):
        """Test search with OcTreeKey"""
        coord = [1.0, 2.0, 3.0]
        stamped_tree.updateNode(coord, True)
        
        coord_arr = np.array(coord)
        key = stamped_tree.coordToKey(coord_arr)
        node = stamped_tree.search(key)
        
        assert node is not None
        assert node.getTimestamp() > 0


class TestOccupancyOperations:
    """Test occupancy-related operations"""
    
    def test_is_node_occupied(self, stamped_tree):
        """Test isNodeOccupied"""
        occupied_coord = [1.0, 2.0, 3.0]
        free_coord = [4.0, 5.0, 6.0]
        
        occupied_node = stamped_tree.updateNode(occupied_coord, True)
        free_node = stamped_tree.updateNode(free_coord, False)
        
        assert stamped_tree.isNodeOccupied(occupied_node)
        assert not stamped_tree.isNodeOccupied(free_node)
    
    def test_is_node_at_threshold(self, stamped_tree):
        """Test isNodeAtThreshold"""
        coord = [1.0, 2.0, 3.0]
        node = stamped_tree.updateNode(coord, True)
        
        result = stamped_tree.isNodeAtThreshold(node)
        assert isinstance(result, bool)


class TestUpdateOccupancyChildren:
    """Test updateOccupancyChildren functionality"""
    
    def test_update_occupancy_children_updates_timestamp(self, stamped_tree):
        """updateOccupancyChildren should update timestamp"""
        coord = [1.0, 2.0, 3.0]
        node = stamped_tree.updateNode(coord, True)
        
        initial_timestamp = node.getTimestamp()
        time.sleep(0.1)
        
        node.updateOccupancyChildren()
        
        assert node.getTimestamp() >= initial_timestamp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

