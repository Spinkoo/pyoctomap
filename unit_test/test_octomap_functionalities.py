#!/usr/bin/env python3
"""
Test script for octomap wrapper functionalities.
Tests both newly added functions (computeChildIdx, computeIndexKey) and existing functionality.
"""

import numpy as np
from octomap import octomap

def test_octree_key_functions():
    """Test OcTreeKey creation and basic operations"""
    print("=" * 50)
    print("Testing OcTreeKey Functions")
    print("=" * 50)
    
    # Create OcTreeKey objects (using new constructor)
    key1 = octomap.OcTreeKey(100, 200, 300)
    key2 = octomap.OcTreeKey(150, 250, 350)
    
    print(f"Key1: [{key1[0]}, {key1[1]}, {key1[2]}]")
    print(f"Key2: [{key2[0]}, {key2[1]}, {key2[2]}]")
    
    # Test computeIndexKey - this is a method of OcTreeKey class
    try:
        level = 2
        index_key = key1.computeIndexKey(level, key2)
        print(f"Index Key (level {level}): [{index_key[0]}, {index_key[1]}, {index_key[2]}]")
    except Exception as e:
        print(f"Error in computeIndexKey: {e}")
    
    # Test computeChildIdx - this is a method of OcTreeKey class
    try:
        depth = 3
        child_idx = key1.computeChildIdx(key2, depth)
        print(f"Child Index (depth {depth}): {child_idx}")
    except Exception as e:
        print(f"Error in computeChildIdx: {e}")

def test_octree_basic_operations():
    """Test basic OcTree operations"""
    print("\n" + "=" * 50)
    print("Testing OcTree Basic Operations")
    print("=" * 50)
    
    # Create an OcTree with resolution 0.1
    octree = octomap.OcTree(0.1)
    print(f"Created OcTree with resolution: {octree.getResolution()}")
    print(f"Tree depth: {octree.getTreeDepth()}")
    print(f"Tree type: {octree.getTreeType()}")
    
    # Test coordinate to key conversion
    coord = np.array([1.5, 2.5, 3.5])
    key = octree.coordToKey(coord)
    print(f"Coordinate {coord} -> Key: [{key[0]}, {key[1]}, {key[2]}]")
    
    # Test key to coordinate conversion
    coord_back = octree.keyToCoord(key)
    print(f"Key [{key[0]}, {key[1]}, {key[2]}] -> Coordinate: {coord_back}")
    
    # Test search
    node = octree.search(coord)
    if node:
        print(f"Found node at coordinate {coord}")
    else:
        print(f"No node found at coordinate {coord}")

def test_point_cloud_operations():
    """Test point cloud insertion and operations"""
    print("\n" + "=" * 50)
    print("Testing Point Cloud Operations")
    print("=" * 50)
    
    octree = octomap.OcTree(0.1)
    
    # Create a simple point cloud (2D array as required)
    points = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0]
    ])
    
    # Sensor origin
    origin = np.array([0.0, 0.0, 0.0])
    
    print(f"Inserting {len(points)} points into octree...")
    try:
        octree.insertPointCloud(points, origin)
        print("Point cloud insertion successful!")
    except Exception as e:
        print(f"Error inserting point cloud: {e}")
        return octree
    
    # Test some queries
    test_point = np.array([1.0, 1.0, 1.0])
    node = octree.search(test_point)
    if node:
        print(f"Node found at {test_point}")
        print(f"Node occupancy: {node.getOccupancy()}")
        print(f"Node log odds: {node.getLogOdds()}")
        
        # Test occupancy check
        try:
            is_occupied = octree.isNodeOccupied(node)
            print(f"Node is occupied: {is_occupied}")
        except Exception as e:
            print(f"Error checking occupancy: {e}")
    else:
        print(f"No node found at {test_point}")
    
    return octree

def test_bounding_box_operations(octree):
    """Test bounding box operations"""
    print("\n" + "=" * 50)
    print("Testing Bounding Box Operations")
    print("=" * 50)
    
    # Set bounding box
    bbx_min = np.array([0.0, 0.0, 0.0])
    bbx_max = np.array([5.0, 5.0, 5.0])
    
    octree.setBBXMin(bbx_min)
    octree.setBBXMax(bbx_max)
    octree.useBBXLimit(True)
    
    print(f"Set BBX: min={bbx_min}, max={bbx_max}")
    
    # Get bounding box info
    try:
        bounds = octree.getBBXBounds()
        center = octree.getBBXCenter()
        min_val = octree.getBBXMin()
        max_val = octree.getBBXMax()
        
        print(f"BBX Bounds: {bounds}")
        print(f"BBX Center: {center}")
        print(f"BBX Min: {min_val}")
        print(f"BBX Max: {max_val}")
    except Exception as e:
        print(f"Error getting BBX info: {e}")
    
    # Test if point is in BBX
    test_points = [
        np.array([2.5, 2.5, 2.5]),  # Inside
        np.array([10.0, 10.0, 10.0])  # Outside
    ]
    
    for point in test_points:
        try:
            in_bbx = octree.inBBX(point)
            print(f"Point {point} in BBX: {in_bbx}")
        except Exception as e:
            print(f"Error checking BBX for {point}: {e}")

def test_tree_statistics(octree):
    """Test tree statistics and information"""
    print("\n" + "=" * 50)
    print("Testing Tree Statistics")
    print("=" * 50)
    
    try:
        print(f"Number of nodes: {octree.calcNumNodes()}")
        print(f"Number of leaf nodes: {octree.getNumLeafNodes()}")
        print(f"Tree size: {octree.size()}")
        print(f"Memory usage: {octree.memoryUsage()}")
        print(f"Memory usage per node: {octree.memoryUsageNode()}")
        print(f"Volume: {octree.volume()}")
        
        # Get metric information
        metric_size = octree.getMetricSize()
        metric_min = octree.getMetricMin()
        metric_max = octree.getMetricMax()
        
        print(f"Metric size: {metric_size}")
        print(f"Metric min: {metric_min}")
        print(f"Metric max: {metric_max}")
        
    except Exception as e:
        print(f"Error getting tree statistics: {e}")

def test_iterators(octree):
    """Test tree iterators"""
    print("\n" + "=" * 50)
    print("Testing Tree Iterators")
    print("=" * 50)
    
    try:
        # Test leaf iterator
        print("Testing leaf iterator...")
        leaf_count = 0
        for leaf in octree.begin_leafs(maxDepth=3):
            coord = leaf.getCoordinate()
            size = leaf.getSize()
            print(f"Leaf {leaf_count}: coord={coord}, size={size}")
            leaf_count += 1
            if leaf_count >= 5:  # Limit output
                break
        
        print(f"Processed {leaf_count} leaves")
        
    except Exception as e:
        print(f"Error with iterators: {e}")

def test_advanced_operations(octree):
    """Test advanced operations"""
    print("\n" + "=" * 50)
    print("Testing Advanced Operations")
    print("=" * 50)
    
    # Test ray casting
    try:
        origin = np.array([0.0, 0.0, 0.0])
        direction = np.array([1.0, 1.0, 1.0])
        end = np.array([0.0, 0.0, 0.0])  # Will be filled by castRay
        
        hit = octree.castRay(origin, direction, end)
        print(f"Ray cast from {origin} in direction {direction}")
        print(f"Hit: {hit}, End point: {end}")
        
    except Exception as e:
        print(f"Error with ray casting: {e}")
    
    # Test node updates
    try:
        test_coord = np.array([4.0, 4.0, 4.0])
        node = octree.updateNode(test_coord, True)  # Mark as occupied
        if node:
            print(f"Updated node at {test_coord}")
            print(f"New occupancy: {node.getOccupancy()}")
        
    except Exception as e:
        print(f"Error updating node: {e}")

def main():
    """Main test function"""
    print("OctoMap Wrapper Functionality Test")
    print("=" * 60)
    
    # Test OcTreeKey functions (new functionality)
    test_octree_key_functions()
    
    # Test basic OcTree operations
    test_octree_basic_operations()
    
    # Test point cloud operations
    octree = test_point_cloud_operations()
    
    # Test bounding box operations
    test_bounding_box_operations(octree)
    
    # Test tree statistics
    test_tree_statistics(octree)
    
    # Test iterators
    test_iterators(octree)
    
    # Test advanced operations
    test_advanced_operations(octree)
    
    print("\n" + "=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    main()