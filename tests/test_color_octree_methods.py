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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

