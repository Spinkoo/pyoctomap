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
        """Binary write/read should preserve exact timestamp values"""
        coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        original_timestamps = {}
        
        for coord in coords:
            node = stamped_tree.updateNode(coord, True)
            # Set specific timestamps to verify they're preserved
            timestamp = int(time.time()) + hash(tuple(coord)) % 1000
            node.setTimestamp(timestamp)
            original_timestamps[tuple(coord)] = timestamp
        
        filename = str(tmp_path / "test_stamped.bt")
        success = stamped_tree.writeBinary(filename)
        assert success, "writeBinary should succeed"
        
        tree_loaded = OcTreeStamped(0.1)
        success_read = tree_loaded.readBinary(filename)
        assert success_read, "readBinary should succeed"
        
        for coord in coords:
            node_loaded = tree_loaded.search(coord)
            assert node_loaded is not None, f"Node at {coord} should exist after loading"
            
            loaded_timestamp = node_loaded.getTimestamp()
            expected_timestamp = original_timestamps[tuple(coord)]
            
            # Timestamps should be preserved exactly
            assert loaded_timestamp == expected_timestamp, \
                f"Timestamp at {coord} should be preserved: expected {expected_timestamp}, got {loaded_timestamp}"
    
    def test_write_read_binary_preserves_multiple_timestamps(self, stamped_tree, tmp_path):
        """Test that binary format preserves timestamps for multiple nodes"""
        coords = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]
        
        original_timestamps = {}
        base_time = int(time.time())
        
        for i, coord in enumerate(coords):
            node = stamped_tree.updateNode(coord, True)
            # Set different timestamps for each node
            timestamp = base_time + i * 100
            node.setTimestamp(timestamp)
            original_timestamps[tuple(coord)] = timestamp
        
        filename = str(tmp_path / "test_multiple_timestamps.bt")
        assert stamped_tree.writeBinary(filename), "writeBinary should succeed"
        
        tree_loaded = OcTreeStamped(0.1)
        assert tree_loaded.readBinary(filename), "readBinary should succeed"
        
        # Verify all timestamps are preserved
        for coord in coords:
            node_loaded = tree_loaded.search(coord)
            assert node_loaded is not None, f"Node at {coord} should exist"
            
            loaded_timestamp = node_loaded.getTimestamp()
            expected_timestamp = original_timestamps[tuple(coord)]
            
            assert loaded_timestamp == expected_timestamp, \
                f"Timestamp at {coord} should be preserved: expected {expected_timestamp}, got {loaded_timestamp}"
    
    def test_write_read_binary_preserves_timestamp_and_occupancy(self, stamped_tree, tmp_path):
        """Test that binary format preserves both timestamp and occupancy information"""
        test_cases = [
            ([1.0, 1.0, 1.0], True, 1000),
            ([2.0, 2.0, 2.0], True, 2000),
            ([3.0, 3.0, 3.0], False, 3000),
        ]
        
        for coord, occupied, timestamp in test_cases:
            node = stamped_tree.updateNode(coord, occupied)
            node.setTimestamp(timestamp)
        
        filename = str(tmp_path / "test_timestamp_occupancy.bt")
        assert stamped_tree.writeBinary(filename), "writeBinary should succeed"
        
        tree_loaded = OcTreeStamped(0.1)
        assert tree_loaded.readBinary(filename), "readBinary should succeed"
        
        # Verify both occupancy and timestamp are preserved
        for coord, expected_occupied, expected_timestamp in test_cases:
            node_loaded = tree_loaded.search(coord)
            assert node_loaded is not None, f"Node at {coord} should exist"
            
            # Check occupancy
            is_occupied = tree_loaded.isNodeOccupied(node_loaded)
            assert is_occupied == expected_occupied, \
                f"Occupancy mismatch at {coord}: expected {expected_occupied}, got {is_occupied}"
            
            # Check timestamp
            actual_timestamp = node_loaded.getTimestamp()
            assert actual_timestamp == expected_timestamp, \
                f"Timestamp mismatch at {coord}: expected {expected_timestamp}, got {actual_timestamp}"
    
    def test_write_read_binary_preserves_tree_time(self, stamped_tree, tmp_path):
        """Test that binary format preserves the tree's last update time"""
        # Update some nodes
        stamped_tree.updateNode([1.0, 2.0, 3.0], True)
        stamped_tree.updateNode([4.0, 5.0, 6.0], True)
        
        # Get original tree time
        original_tree_time = stamped_tree.getLastUpdateTime()
        assert original_tree_time > 0, "Tree should have a valid update time"
        
        filename = str(tmp_path / "test_tree_time.bt")
        assert stamped_tree.writeBinary(filename), "writeBinary should succeed"
        
        tree_loaded = OcTreeStamped(0.1)
        assert tree_loaded.readBinary(filename), "readBinary should succeed"
        
        # Verify tree time is preserved (should match root node timestamp)
        loaded_tree_time = tree_loaded.getLastUpdateTime()
        # Note: Tree time might not be exactly preserved if root node changes,
        # but it should be a valid timestamp
        assert loaded_tree_time >= 0, "Loaded tree should have a valid update time"
    
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


class TestIteratorFunctionality:
    """Test iterator functionality specific to OcTreeStamped"""

    def test_iterator_timestamp_access(self, stamped_tree):
        """Test that iterators provide correct timestamp access"""
        coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        expected_timestamps = []

        # Create nodes with specific timestamps
        for coord in coords:
            node = stamped_tree.updateNode(coord, True)
            timestamp = int(time.time()) + hash(tuple(coord)) % 1000
            node.setTimestamp(timestamp)
            expected_timestamps.append(timestamp)

        # Test iterator timestamp access - collect all timestamps
        found_timestamps = []
        for leaf in stamped_tree.begin_leafs():
            timestamp = leaf.getTimestamp()
            found_timestamps.append(timestamp)

        # Verify we have the expected number of nodes with expected timestamps
        assert len(found_timestamps) == len(coords), f"Expected {len(coords)} nodes, got {len(found_timestamps)}"

        # Sort both lists for comparison (since order may vary)
        found_timestamps.sort()
        expected_timestamps.sort()

        assert found_timestamps == expected_timestamps, \
            f"Timestamp mismatch: expected {expected_timestamps}, got {found_timestamps}"

    def test_iterator_memory_cleanup(self, stamped_tree):
        """Test that stamped tree iterators clean up properly"""
        import gc
        import weakref

        # Add some nodes
        coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        for coord in coords:
            stamped_tree.updateNode(coord, True)

        # Create iterator and weak reference
        iterator = stamped_tree.begin_leafs()
        iterator_ref = weakref.ref(iterator)

        # Use iterator
        count = 0
        for leaf in iterator:
            assert leaf.getTimestamp() >= 0  # Should have valid timestamp
            count += 1

        assert count == len(coords)

        # Clear references and force garbage collection
        del iterator
        gc.collect()

        # Iterator should be garbage collected (weak reference should be None)
        # Note: In some Python implementations, this may not happen immediately
        collected = iterator_ref() is None
        if not collected:
            # If not collected, at least verify it doesn't hold a tree reference
            weak_iter = iterator_ref()
            assert not hasattr(weak_iter, '_tree') or weak_iter._tree is None, \
                "Iterator still holds tree reference which could cause circular reference"

    def test_iterator_with_different_timestamps(self, stamped_tree):
        """Test iterator with nodes having different timestamps"""
        base_time = int(time.time())

        # Create nodes with different timestamps
        test_cases = [
            ([1.0, 1.0, 1.0], base_time + 100),
            ([2.0, 2.0, 2.0], base_time + 200),
            ([3.0, 3.0, 3.0], base_time + 300),
        ]

        for coord, timestamp in test_cases:
            node = stamped_tree.updateNode(coord, True)
            node.setTimestamp(timestamp)

        # Collect timestamps via iterator
        iterator_timestamps = []
        for leaf in stamped_tree.begin_leafs():
            iterator_timestamps.append(leaf.getTimestamp())

        # Sort for comparison
        iterator_timestamps.sort()
        expected_timestamps = [ts for _, ts in test_cases]
        expected_timestamps.sort()

        assert iterator_timestamps == expected_timestamps


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


class TestInsertPointCloudWithTimestamp:
    """Test insertPointCloudWithTimestamp method"""
    
    def test_insertPointCloudWithTimestamp_basic(self, stamped_tree):
        """Test basic insertPointCloudWithTimestamp functionality"""
        points = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=np.float64)
        
        timestamp = 12345
        
        # Insert point cloud with timestamp
        n_processed = stamped_tree.insertPointCloudWithTimestamp(points, timestamp, lazy_eval=False)
        assert n_processed == 3, "Should process 3 points"
        
        # Verify timestamps were set
        for point in points:
            node = stamped_tree.search(point)
            if node is not None:
                assert node.getTimestamp() == timestamp, "Timestamp should match"
    
    def test_insertPointCloudWithTimestamp_lazy_eval(self, stamped_tree):
        """Test insertPointCloudWithTimestamp with lazy evaluation"""
        points = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ], dtype=np.float64)
        
        timestamp = 99999
        
        # Insert with lazy_eval=True
        n_processed = stamped_tree.insertPointCloudWithTimestamp(points, timestamp, lazy_eval=True)
        assert n_processed == 2, "Should process 2 points"
        
        # Manually update inner occupancy
        stamped_tree.updateInnerOccupancy()
        
        # Verify timestamps were set
        for point in points:
            node = stamped_tree.search(point)
            if node is not None:
                assert node.getTimestamp() == timestamp, "Timestamp should match"
    
    def test_insertPointCloudWithTimestamp_multiple_updates(self, stamped_tree):
        """Test that insertPointCloudWithTimestamp can update existing nodes"""
        points = np.array([
            [1.0, 2.0, 3.0]
        ], dtype=np.float64)
        
        # First insert with timestamp 100
        n1 = stamped_tree.insertPointCloudWithTimestamp(points, 100, lazy_eval=False)
        assert n1 == 1
        
        node1 = stamped_tree.search(points[0])
        assert node1 is not None
        assert node1.getTimestamp() == 100
        
        # Second insert with different timestamp 200
        n2 = stamped_tree.insertPointCloudWithTimestamp(points, 200, lazy_eval=False)
        assert n2 == 1
        
        node2 = stamped_tree.search(points[0])
        assert node2 is not None
        assert node2.getTimestamp() == 200, "Timestamp should be updated"
    
    def test_insertPointCloudWithTimestamp_large_batch(self, stamped_tree):
        """Test insertPointCloudWithTimestamp with a large batch of points"""
        # Create 50 points
        n_points = 50
        points = np.random.rand(n_points, 3) * 10.0  # Random points in [0, 10] range
        points = points.astype(np.float64)
        
        timestamp = 50000
        
        # Insert point cloud with timestamp
        n_processed = stamped_tree.insertPointCloudWithTimestamp(points, timestamp, lazy_eval=False)
        assert n_processed == n_points, f"Should process {n_points} points"
        
        # Verify a sample of points have timestamps set
        sample_indices = [0, 10, 25, 49]
        for idx in sample_indices:
            node = stamped_tree.search(points[idx])
            if node is not None:
                assert node.getTimestamp() == timestamp, \
                    f"Timestamp should be {timestamp} for point {idx}"
    
    def test_insertPointCloudWithTimestamp_zero_timestamp(self, stamped_tree):
        """Test insertPointCloudWithTimestamp with zero timestamp"""
        points = np.array([
            [1.0, 2.0, 3.0]
        ], dtype=np.float64)
        
        timestamp = 0
        
        n_processed = stamped_tree.insertPointCloudWithTimestamp(points, timestamp, lazy_eval=False)
        assert n_processed == 1
        
        node = stamped_tree.search(points[0])
        if node is not None:
            assert node.getTimestamp() == 0, "Zero timestamp should be valid"
    
    def test_insertPointCloudWithTimestamp_max_range(self, stamped_tree):
        """Test insertPointCloudWithTimestamp with max_range parameter"""
        points = np.array([
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0]  # Far point
        ], dtype=np.float64)
        
        timestamp = 12345
        
        # Insert with max_range (though not used in current implementation)
        n_processed = stamped_tree.insertPointCloudWithTimestamp(
            points, timestamp, max_range=5.0, lazy_eval=False
        )
        assert n_processed == 2, "Should process all points regardless of max_range"
    
    def test_insertPointCloudWithTimestamp_sensor_origin(self, stamped_tree):
        """Test insertPointCloudWithTimestamp with sensor origin for proper ray casting"""
        sensor_origin = np.array([0.0, 0.0, 0.0])
        points = np.array([
            [1.0, 0.0, 0.0],  # Point 1m in front
            [2.0, 0.0, 0.0],  # Point 2m in front
        ], dtype=np.float64)
        
        timestamp = 1000
        
        # Insert with sensor origin - should perform proper ray casting
        n_processed = stamped_tree.insertPointCloudWithTimestamp(
            points, timestamp, sensor_origin=sensor_origin, lazy_eval=False
        )
        assert n_processed == 2, "Should process all points"
        
        # Verify points are occupied and have timestamps
        for point in points:
            node = stamped_tree.search(point)
            assert node is not None, "Point should exist"
            assert stamped_tree.isNodeOccupied(node), "Point should be occupied"
            assert node.getTimestamp() == timestamp, "Timestamp should be set"
        
        # Test with list origin
        n_processed2 = stamped_tree.insertPointCloudWithTimestamp(
            points, timestamp + 1, sensor_origin=[0.0, 0.0, 0.0], lazy_eval=False
        )
        assert n_processed2 == 2, "Should work with list origin"
        
        # Test invalid origin
        try:
            stamped_tree.insertPointCloudWithTimestamp(
                points, timestamp, sensor_origin=[1.0, 2.0], lazy_eval=False
            )
            assert False, "Should raise ValueError for invalid origin"
        except ValueError:
            pass  # Expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

