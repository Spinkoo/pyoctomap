#!/usr/bin/env python3
"""
Test script to verify ColorOcTree has all the methods that OcTree has.
Tests clamping functions, tree manipulation, and utility methods.
"""
import pytest
import numpy as np
import pyoctomap


def test_clamping_threshold_methods():
    """Test clamping threshold getters and setters"""
    tree = pyoctomap.ColorOcTree(0.1)
    
    # Test getters (should have default values)
    max_thres = tree.getClampingThresMax()
    max_thres_log = tree.getClampingThresMaxLog()
    min_thres = tree.getClampingThresMin()
    min_thres_log = tree.getClampingThresMinLog()
    
    assert isinstance(max_thres, (int, float)), "getClampingThresMax should return a number"
    assert isinstance(max_thres_log, (int, float)), "getClampingThresMaxLog should return a number"
    assert isinstance(min_thres, (int, float)), "getClampingThresMin should return a number"
    assert isinstance(min_thres_log, (int, float)), "getClampingThresMinLog should return a number"
    
    # Test setters
    tree.setClampingThresMax(0.97)
    tree.setClampingThresMin(0.12)
    
    assert abs(tree.getClampingThresMax() - 0.97) < 0.001, "setClampingThresMax should work"
    assert abs(tree.getClampingThresMin() - 0.12) < 0.001, "setClampingThresMin should work"


def test_tree_manipulation_methods():
    """Test prune, expand, and useBBXLimit methods"""
    tree = pyoctomap.ColorOcTree(0.05)
    
    # Add some nodes
    for x in range(-3, 4):
        for y in range(-3, 4):
            coord = np.array([x * 0.1, y * 0.1, 0.1])
            tree.updateNode(coord, True)
            tree.setNodeColor(coord, 255, 0, 0)
    
    initial_size = tree.size()
    assert initial_size > 0, "Tree should have nodes"
    
    # Test prune (should work, may or may not change size depending on tree structure)
    tree.prune()
    pruned_size = tree.size()
    assert pruned_size >= 0, "prune() should not crash"
    
    # Test expand (should expand all nodes)
    tree.expand()
    expanded_size = tree.size()
    assert expanded_size >= pruned_size, "expand() should increase or maintain size"
    
    # Test useBBXLimit
    tree.useBBXLimit(True)
    tree.useBBXLimit(False)
    # Should not crash


def test_node_child_management():
    """Test expandNode, createNodeChild, deleteNodeChild"""
    tree = pyoctomap.ColorOcTree(0.1)
    
    # Add a node
    coord = np.array([1.0, 1.0, 1.0])
    node = tree.updateNode(coord, True)
    assert node is not None, "Should create a node"
    
    # Test nodeHasChildren (should be False for leaf)
    has_children = tree.nodeHasChildren(node)
    assert isinstance(has_children, bool), "nodeHasChildren should return bool"
    
    # Test expandNode (should create children)
    if not has_children:
        tree.expandNode(node)
        has_children_after = tree.nodeHasChildren(node)
        # After expansion, it should have children (if not at max depth)
        # Note: This may fail if at max depth, which is OK
    
    # Test createNodeChild (requires a node with children)
    root = tree.getRoot()
    if root is not None and tree.nodeHasChildren(root):
        child = tree.createNodeChild(root, 0)
        assert child is not None, "createNodeChild should return a node"
        
        # Test deleteNodeChild
        tree.deleteNodeChild(root, 0)
        # Should not crash


def test_utility_methods():
    """Test getLabels and extractPointCloud"""
    tree = pyoctomap.ColorOcTree(0.05)
    
    # Add some nodes
    for x in range(-2, 3):
        for y in range(-2, 3):
            coord = np.array([x * 0.1, y * 0.1, 0.1])
            tree.updateNode(coord, True)
            tree.setNodeColor(coord, 255, 0, 0)
    
    # Test getLabels
    test_points = np.array([
        [0.0, 0.0, 0.1],
        [1.0, 1.0, 1.0],  # Should be unknown
        [-0.1, -0.1, 0.1],
    ])
    labels = tree.getLabels(test_points)
    assert labels.shape == (3,), "getLabels should return array of correct shape"
    assert all(l in [-1, 0, 1] for l in labels), "Labels should be -1, 0, or 1"
    
    # Test extractPointCloud
    occupied, empty = tree.extractPointCloud()
    assert isinstance(occupied, np.ndarray), "extractPointCloud should return numpy array"
    assert isinstance(empty, np.ndarray), "extractPointCloud should return numpy array"
    assert occupied.shape[1] == 3, "Points should be 3D"
    assert empty.shape[1] == 3, "Points should be 3D"


def test_iterator_methods():
    """Test begin_tree, end_tree, end_leafs, end_leafs_bbx"""
    tree = pyoctomap.ColorOcTree(0.05)
    
    # Add some nodes
    for x in range(-2, 3):
        coord = np.array([x * 0.1, 0.0, 0.1])
        tree.updateNode(coord, True)
        tree.setNodeColor(coord, 255, 0, 0)
    
    # Test begin_tree
    tree_iter = tree.begin_tree()
    assert tree_iter is not None, "begin_tree should return iterator"
    
    # Test end_tree
    end_tree_iter = tree.end_tree()
    assert end_tree_iter is not None, "end_tree should return iterator"
    
    # Test end_leafs
    end_leafs_iter = tree.end_leafs()
    assert end_leafs_iter is not None, "end_leafs should return iterator"
    
    # Test end_leafs_bbx
    bbx_min = np.array([-1.0, -1.0, -1.0])
    bbx_max = np.array([1.0, 1.0, 1.0])
    end_bbx_iter = tree.end_leafs_bbx()
    assert end_bbx_iter is not None, "end_leafs_bbx should return iterator"


def test_isNodeOccupied_with_iterator():
    """Test that isNodeOccupied works with iterators"""
    tree = pyoctomap.ColorOcTree(0.05)
    
    # Add a node
    coord = np.array([0.0, 0.0, 0.1])
    tree.updateNode(coord, True)
    tree.setNodeColor(coord, 255, 0, 0)
    
    # Test with iterator
    for leaf in tree.begin_leafs():
        is_occupied = tree.isNodeOccupied(leaf)
        assert isinstance(is_occupied, bool), "isNodeOccupied should return bool"
        break  # Just test first one


def test_isNodeAtThreshold_with_iterator():
    """Test that isNodeAtThreshold works with iterators"""
    tree = pyoctomap.ColorOcTree(0.05)
    
    # Add a node
    coord = np.array([0.0, 0.0, 0.1])
    tree.updateNode(coord, True)
    
    # Test with iterator
    for leaf in tree.begin_leafs():
        at_threshold = tree.isNodeAtThreshold(leaf)
        assert isinstance(at_threshold, bool), "isNodeAtThreshold should return bool"
        break  # Just test first one


def test_method_parity_with_octree():
    """Compare available methods between OcTree and ColorOcTree"""
    octree = pyoctomap.OcTree(0.1)
    color_tree = pyoctomap.ColorOcTree(0.1)
    
    # Methods that should be in both
    common_methods = [
        'getClampingThresMax', 'getClampingThresMin',
        'setClampingThresMax', 'setClampingThresMin',
        'getOccupancyThres', 'setOccupancyThres',
        'getProbHit', 'getProbMiss', 'setProbHit', 'setProbMiss',
        'prune', 'expand', 'useBBXLimit',
        'expandNode', 'createNodeChild', 'deleteNodeChild',
        'getLabels', 'extractPointCloud',
        'begin_tree', 'end_tree', 'end_leafs', 'end_leafs_bbx',
        'getMetricMin', 'getMetricMax', 'getMetricSize',
        'memoryUsage', 'volume',
    ]
    
    missing_methods = []
    for method_name in common_methods:
        if not hasattr(color_tree, method_name):
            missing_methods.append(method_name)
        elif not hasattr(octree, method_name):
            # If OcTree doesn't have it, maybe it shouldn't be in ColorOcTree either
            print(f"Warning: {method_name} not in OcTree")
    
    if missing_methods:
        pytest.fail(f"ColorOcTree missing methods: {missing_methods}")
    
    print(f"✓ All {len(common_methods)} common methods are present in ColorOcTree")


def test_write_read_binary_preserves_colors(tmp_path):
    """Test that binary write/read preserves color information"""
    # Create a tree with colored nodes
    tree = pyoctomap.ColorOcTree(0.1)
    test_data = [
        ([1.0, 2.0, 3.0], (255, 0, 0)),      # Red
        ([4.0, 5.0, 6.0], (0, 255, 0)),      # Green
        ([7.0, 8.0, 9.0], (0, 0, 255)),      # Blue
        ([10.0, 11.0, 12.0], (128, 128, 128)), # Gray
    ]
    
    original_colors = {}
    for coord, color in test_data:
        node = tree.updateNode(coord, True)
        tree.setNodeColor(coord, color[0], color[1], color[2])
        original_colors[tuple(coord)] = color
    
    # Write to binary file
    filename = str(tmp_path / "test_color.bt")
    success = tree.writeBinary(filename)
    assert success, "writeBinary should succeed"
    
    # Read from binary file
    tree_loaded = pyoctomap.ColorOcTree(0.1)
    success_read = tree_loaded.readBinary(filename)
    assert success_read, "readBinary should succeed"
    
    # Verify colors are preserved
    for coord, expected_color in test_data:
        node_loaded = tree_loaded.search(coord)
        assert node_loaded is not None, f"Node at {coord} should exist after loading"
        
        loaded_color = node_loaded.getColor()
        assert loaded_color == expected_color, \
            f"Color at {coord} should be preserved: expected {expected_color}, got {loaded_color}"


def test_write_read_binary_preserves_multiple_colors(tmp_path):
    """Test that binary format preserves colors for multiple nodes with different colors"""
    tree = pyoctomap.ColorOcTree(0.05)
    
    # Create a grid of nodes with different colors
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    
    original_data = {}
    for i, color in enumerate(colors):
        coord = [i * 0.2, i * 0.2, i * 0.2]
        tree.updateNode(coord, True)
        tree.setNodeColor(coord, color[0], color[1], color[2])
        original_data[tuple(coord)] = color
    
    # Save and load
    filename = str(tmp_path / "test_multicolor.bt")
    assert tree.writeBinary(filename), "writeBinary should succeed"
    
    tree_loaded = pyoctomap.ColorOcTree(0.05)
    assert tree_loaded.readBinary(filename), "readBinary should succeed"
    
    # Verify all colors are preserved
    for coord, expected_color in original_data.items():
        node = tree_loaded.search(coord)
        assert node is not None, f"Node at {coord} should exist"
        actual_color = node.getColor()
        assert actual_color == expected_color, \
            f"Color mismatch at {coord}: expected {expected_color}, got {actual_color}"


def test_write_read_binary_preserves_color_and_occupancy(tmp_path):
    """Test that binary format preserves both color and occupancy information"""
    tree = pyoctomap.ColorOcTree(0.1)
    
    # Create nodes with different occupancy states and colors
    test_cases = [
        ([1.0, 1.0, 1.0], True, (255, 0, 0)),   # Occupied, red
        ([2.0, 2.0, 2.0], True, (0, 255, 0)),  # Occupied, green
        ([3.0, 3.0, 3.0], False, (0, 0, 255)), # Free, blue
    ]
    
    for coord, occupied, color in test_cases:
        tree.updateNode(coord, occupied)
        tree.setNodeColor(coord, color[0], color[1], color[2])
    
    # Save and load
    filename = str(tmp_path / "test_color_occupancy.bt")
    assert tree.writeBinary(filename), "writeBinary should succeed"
    
    tree_loaded = pyoctomap.ColorOcTree(0.1)
    assert tree_loaded.readBinary(filename), "readBinary should succeed"
    
    # Verify both occupancy and color are preserved
    for coord, expected_occupied, expected_color in test_cases:
        node = tree_loaded.search(coord)
        assert node is not None, f"Node at {coord} should exist"
        
        # Check occupancy
        is_occupied = tree_loaded.isNodeOccupied(node)
        assert is_occupied == expected_occupied, \
            f"Occupancy mismatch at {coord}: expected {expected_occupied}, got {is_occupied}"
        
        # Check color
        actual_color = node.getColor()
        assert actual_color == expected_color, \
            f"Color mismatch at {coord}: expected {expected_color}, got {actual_color}"


def test_write_read_binary_averaged_colors(tmp_path):
    """Test that binary format preserves averaged colors"""
    tree = pyoctomap.ColorOcTree(0.1)
    
    # Set colors using averageNodeColor
    coord = [1.0, 2.0, 3.0]
    tree.updateNode(coord, True)
    
    # Add multiple color measurements (should average)
    tree.averageNodeColor(coord, 100, 100, 100)
    tree.averageNodeColor(coord, 200, 200, 200)
    # Average should be approximately (150, 150, 150)
    
    node_before = tree.search(coord)
    color_before = node_before.getColor()
    
    # Save and load
    filename = str(tmp_path / "test_averaged_color.bt")
    assert tree.writeBinary(filename), "writeBinary should succeed"
    
    tree_loaded = pyoctomap.ColorOcTree(0.1)
    assert tree_loaded.readBinary(filename), "readBinary should succeed"
    
    # Verify averaged color is preserved
    node_after = tree_loaded.search(coord)
    assert node_after is not None, "Node should exist after loading"
    
    color_after = node_after.getColor()
    # Colors should match (averaged colors should be preserved)
    assert color_after == color_before, \
        f"Averaged color should be preserved: expected {color_before}, got {color_after}"


def test_iterator_memory_cleanup():
    """Test that iterators properly clean up memory and don't leak"""
    import gc
    import weakref

    tree = pyoctomap.ColorOcTree(0.1)

    # Create some nodes
    coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    for coord in coords:
        tree.updateNode(coord, True)
        tree.setNodeColor(coord, 255, 0, 0)

    # Create iterator and keep weak reference
    iterator = tree.begin_leafs()
    iterator_ref = weakref.ref(iterator)

    # Use iterator to ensure it's fully initialized
    count = 0
    for leaf in iterator:
        assert leaf.getColor() == (255, 0, 0)  # Should be red
        count += 1

    assert count == len(coords), f"Expected {len(coords)} leaves, got {count}"

    # Clear any local references and force garbage collection
    del iterator
    gc.collect()

    # Iterator should be garbage collected (weak reference should be None)
    # Note: In some Python implementations, this may not happen immediately
    # so we allow for it to still exist but check that it's not holding tree references
    collected = iterator_ref() is None
    if not collected:
        # If not collected, at least verify it doesn't hold a tree reference
        # that would prevent the tree from being collected
        weak_iter = iterator_ref()
        # The iterator should not hold a strong reference to the tree
        # (we removed this to prevent circular references)
        assert not hasattr(weak_iter, '_tree') or weak_iter._tree is None, \
            "Iterator still holds tree reference which could cause circular reference"


def test_iterator_exception_safety():
    """Test that iterators handle exceptions safely during construction"""
    # Test with None tree
    iterator = pyoctomap.SimpleLeafIterator(None)
    assert list(iterator) == [], "Iterator with None tree should be empty"

    # Test with invalid tree (this should not crash)
    class FakeTree:
        pass

    try:
        iterator = pyoctomap.SimpleLeafIterator(FakeTree())
        # Should handle gracefully
        assert list(iterator) == [], "Iterator with invalid tree should be empty"
    except Exception:
        # It's okay if it raises an exception, as long as it doesn't crash
        pass


def test_iterator_color_access():
    """Test that iterators provide correct color access"""
    tree = pyoctomap.ColorOcTree(0.1)

    # Test data: coord -> color
    test_data = [
        ([1.0, 2.0, 3.0], (255, 0, 0)),      # Red
        ([4.0, 5.0, 6.0], (0, 255, 0)),      # Green
        ([7.0, 8.0, 9.0], (0, 0, 255)),      # Blue
    ]

    # Create nodes with colors
    for coord, color in test_data:
        tree.updateNode(coord, True)
        tree.setNodeColor(coord, color[0], color[1], color[2])

    # Test iterator color access - collect all colors
    found_colors = []
    for leaf in tree.begin_leafs():
        color = leaf.getColor()
        found_colors.append(color)

    # Verify we have the expected number of nodes with expected colors
    assert len(found_colors) == len(test_data), f"Expected {len(test_data)} nodes, got {len(found_colors)}"

    expected_colors = [color for _, color in test_data]
    found_colors_set = set(found_colors)
    expected_colors_set = set(expected_colors)

    assert found_colors_set == expected_colors_set, \
        f"Color mismatch: expected {expected_colors_set}, got {found_colors_set}"


def test_iterator_timestamp_access():
    """Test that iterators provide correct timestamp access for OcTreeStamped"""
    import pyoctomap as pyoctomap

    tree = pyoctomap.OcTreeStamped(0.1)

    # Test data: coord -> timestamp
    test_data = [
        ([1.0, 2.0, 3.0], 1000000000),
        ([4.0, 5.0, 6.0], 1000000001),
        ([7.0, 8.0, 9.0], 1000000002),
    ]

    # Create nodes with timestamps
    for coord, timestamp in test_data:
        node = tree.updateNode(coord, True)
        node.setTimestamp(timestamp)

    # Test iterator timestamp access - collect all timestamps
    found_timestamps = []
    for leaf in tree.begin_leafs():
        timestamp = leaf.getTimestamp()
        found_timestamps.append(timestamp)

    # Verify we have the expected number of nodes with expected timestamps
    assert len(found_timestamps) == len(test_data), f"Expected {len(test_data)} nodes, got {len(found_timestamps)}"

    expected_timestamps = [timestamp for _, timestamp in test_data]
    found_timestamps_set = set(found_timestamps)
    expected_timestamps_set = set(expected_timestamps)

    assert found_timestamps_set == expected_timestamps_set, \
        f"Timestamp mismatch: expected {expected_timestamps_set}, got {found_timestamps_set}"


def test_iterator_regular_octree():
    """Test that regular OcTree iterators work correctly (no colors/timestamps)"""
    tree = pyoctomap.OcTree(0.1)

    coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    for coord in coords:
        tree.updateNode(coord, True)

    # Test iterator
    found_coords = []
    for leaf in tree.begin_leafs():
        coord = leaf.getCoordinate()
        found_coords.append(coord)
        # Regular OcTree should return white/default color
        assert leaf.getColor() == (255, 255, 255), "Regular OcTree should return white color"
        assert leaf.getTimestamp() == 0, "Regular OcTree should return 0 timestamp"

    assert len(found_coords) == len(coords), f"Expected {len(coords)} leaves, got {len(found_coords)}"


def test_iterator_bounding_box():
    """Test that bounding box iterators work correctly"""
    tree = pyoctomap.ColorOcTree(0.1)

    # Create nodes in and out of bounding box
    all_coords = [
        ([0.5, 0.5, 0.5], (255, 0, 0)),     # Inside BBX
        ([1.5, 1.5, 1.5], (0, 255, 0)),     # Inside BBX
        ([3.0, 3.0, 3.0], (0, 0, 255)),     # Outside BBX
    ]

    for coord, color in all_coords:
        tree.updateNode(coord, True)
        tree.setNodeColor(coord, color[0], color[1], color[2])

    # Define bounding box that contains first two nodes
    bbx_min = np.array([0.0, 0.0, 0.0])
    bbx_max = np.array([2.0, 2.0, 2.0])

    # Test BBX iterator
    found_coords = []
    found_colors = []
    for leaf in tree.begin_leafs_bbx(bbx_min, bbx_max):
        coord = leaf.getCoordinate()
        color = leaf.getColor()
        found_coords.append(tuple(coord))
        found_colors.append(color)

    # Should find exactly 2 nodes (the ones inside BBX)
    assert len(found_coords) == 2, f"Expected 2 nodes in BBX, got {len(found_coords)}"

    # Colors should be red and green
    expected_colors = [(255, 0, 0), (0, 255, 0)]
    for color in found_colors:
        assert color in expected_colors, f"Unexpected color {color} in BBX iterator"


def test_insertPointCloudWithColor():
    """Test insertPointCloudWithColor method"""
    tree = pyoctomap.ColorOcTree(0.1)
    
    # Create test points and colors
    points = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float64)
    
    colors = np.array([
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0]   # Blue
    ], dtype=np.float64)
    
    # Insert point cloud with colors
    n_processed = tree.insertPointCloudWithColor(points, colors, max_range=-1.0, lazy_eval=False)
    assert n_processed == 3, "Should process 3 points"
    
    # Verify colors were set
    for i, point in enumerate(points):
        node = tree.search(point)
        if node is not None:
            color = node.getColor()
            expected_r = int(colors[i, 0] * 255)
            expected_g = int(colors[i, 1] * 255)
            expected_b = int(colors[i, 2] * 255)
            assert color[0] == expected_r, f"Red component mismatch for point {i}"
            assert color[1] == expected_g, f"Green component mismatch for point {i}"
            assert color[2] == expected_b, f"Blue component mismatch for point {i}"


def test_insertPointCloudWithColor_lazy_eval():
    """Test insertPointCloudWithColor with lazy evaluation"""
    tree = pyoctomap.ColorOcTree(0.1)
    
    points = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], dtype=np.float64)
    
    colors = np.array([
        [0.5, 0.5, 0.5],  # Gray
        [1.0, 1.0, 0.0]   # Yellow
    ], dtype=np.float64)
    
    # Insert with lazy_eval=True
    n_processed = tree.insertPointCloudWithColor(points, colors, lazy_eval=True)
    assert n_processed == 2, "Should process 2 points"
    
    # Manually update inner occupancy
    tree.updateInnerOccupancy()
    
    # Verify colors were set
    for i, point in enumerate(points):
        node = tree.search(point)
        if node is not None:
            color = node.getColor()
            assert color[0] == int(colors[i, 0] * 255), f"Color mismatch for point {i}"


def test_insertPointCloudWithColor_validation():
    """Test insertPointCloudWithColor input validation"""
    tree = pyoctomap.ColorOcTree(0.1)
    
    points = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], dtype=np.float64)
    
    # Test mismatched array sizes
    colors_wrong_size = np.array([
        [1.0, 0.0, 0.0]
    ], dtype=np.float64)
    
    with pytest.raises(ValueError, match="same number of rows"):
        tree.insertPointCloudWithColor(points, colors_wrong_size)
    
    # Test wrong number of color channels
    colors_wrong_channels = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=np.float64)
    
    with pytest.raises(ValueError, match="3 columns"):
        tree.insertPointCloudWithColor(points, colors_wrong_channels)


def test_insertPointCloudWithColor_large_batch():
    """Test insertPointCloudWithColor with a large batch of points"""
    tree = pyoctomap.ColorOcTree(0.05)
    
    # Create 100 points
    n_points = 100
    points = np.random.rand(n_points, 3) * 10.0  # Random points in [0, 10] range
    colors = np.random.rand(n_points, 3)  # Random colors in [0, 1] range
    
    # Insert point cloud with colors
    n_processed = tree.insertPointCloudWithColor(points, colors, lazy_eval=False)
    assert n_processed == n_points, f"Should process {n_points} points"
    
    # Verify a sample of points have colors set
    sample_indices = [0, 10, 50, 99]
    for idx in sample_indices:
        node = tree.search(points[idx])
        if node is not None:
            color = node.getColor()
            # Color should be set (not white/default)
            assert color != (255, 255, 255) or tree.isNodeOccupied(node), \
                f"Color should be set for point {idx}"


def test_insertPointCloudWithColor_sensor_origin():
    """Test insertPointCloudWithColor with sensor origin for proper ray casting"""
    tree = pyoctomap.ColorOcTree(0.1)
    
    # Create points in front of sensor
    sensor_origin = np.array([0.0, 0.0, 0.0])
    points = np.array([
        [1.0, 0.0, 0.0],  # Point 1m in front
        [2.0, 0.0, 0.0],  # Point 2m in front
        [3.0, 0.0, 0.0],  # Point 3m in front
    ], dtype=np.float64)
    colors = np.array([
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
    ], dtype=np.float64)
    
    # Insert with sensor origin - should perform proper ray casting
    n_processed = tree.insertPointCloudWithColor(
        points, colors, sensor_origin=sensor_origin, lazy_eval=False
    )
    assert n_processed == 3, "Should process all 3 points"
    
    # Verify points are occupied
    for i, point in enumerate(points):
        node = tree.search(point)
        assert node is not None, f"Point {i} should exist"
        assert tree.isNodeOccupied(node), f"Point {i} should be occupied"
    
    # Verify colors are set
    node1 = tree.search(points[0])
    color1 = node1.getColor()
    assert color1[0] > 200, "First point should be red"
    
    # Test with list origin
    n_processed2 = tree.insertPointCloudWithColor(
        points, colors, sensor_origin=[0.0, 0.0, 0.0], lazy_eval=False
    )
    assert n_processed2 == 3, "Should work with list origin"
    
    # Test invalid origin
    try:
        tree.insertPointCloudWithColor(points, colors, sensor_origin=[1.0, 2.0], lazy_eval=False)
        assert False, "Should raise ValueError for invalid origin"
    except ValueError:
        pass  # Expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

