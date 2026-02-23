#!/usr/bin/env python3
"""
Standalone test script for PyOctoMap iterator functionality and memory management.

This script tests:
- Iterator color and timestamp access
- Memory cleanup and garbage collection
- Exception safety
- Different tree types

Run with: python test_iterator_functionality.py
"""

import sys
import os
import gc
import weakref
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pyoctomap
except ImportError as e:
    print(f"❌ Failed to import pyoctomap: {e}")
    sys.exit(1)


def test_color_iterator():
    """Test ColorOcTree iterator functionality"""
    print("🟡 Testing ColorOcTree iterator...")

    tree = pyoctomap.ColorOcTree(0.1)

    # Create nodes with different colors
    test_data = [
        ([1.0, 2.0, 3.0], (255, 0, 0)),      # Red
        ([4.0, 5.0, 6.0], (0, 255, 0)),      # Green
        ([7.0, 8.0, 9.0], (0, 0, 255)),      # Blue
    ]

    # Add nodes to tree
    for coord, color in test_data:
        tree.updateNode(coord, True)
        tree.setNodeColor(coord, color[0], color[1], color[2])

    # Test iterator color access
    found_colors = {}
    for leaf in tree.begin_leafs():
        coord = tuple(leaf.getCoordinate())
        color = leaf.getColor()
        found_colors[coord] = color

    # Verify colors
    for coord, expected_color in test_data:
        coord_tuple = tuple(coord)
        assert coord_tuple in found_colors, f"Missing coordinate {coord}"
        assert found_colors[coord_tuple] == expected_color, \
            f"Color mismatch at {coord}: expected {expected_color}, got {found_colors[coord_tuple]}"

    print("    ✅ Color access working")
    return tree


def test_timestamp_iterator():
    """Test OcTreeStamped iterator functionality"""
    print("🕒 Testing OcTreeStamped iterator...")

    tree = pyoctomap.OcTreeStamped(0.1)

    # Create nodes with different timestamps
    base_time = 1000000000  # Fixed base time for testing
    test_data = [
        ([1.0, 2.0, 3.0], base_time + 1000),
        ([4.0, 5.0, 6.0], base_time + 2000),
        ([7.0, 8.0, 9.0], base_time + 3000),
    ]

    # Add nodes to tree
    for coord, timestamp in test_data:
        node = tree.updateNode(coord, True)
        node.setTimestamp(timestamp)

    # Test iterator timestamp access
    found_timestamps = {}
    for leaf in tree.begin_leafs():
        coord = tuple(leaf.getCoordinate())
        timestamp = leaf.getTimestamp()
        found_timestamps[coord] = timestamp

    # Verify timestamps
    for coord, expected_timestamp in test_data:
        coord_tuple = tuple(coord)
        assert coord_tuple in found_timestamps, f"Missing coordinate {coord}"
        assert found_timestamps[coord_tuple] == expected_timestamp, \
            f"Timestamp mismatch at {coord}: expected {expected_timestamp}, got {found_timestamps[coord_tuple]}"

    print("    ✅ Timestamp access working")
    return tree


def test_regular_octree_iterator():
    """Test regular OcTree iterator (should return defaults)"""
    print("🌳 Testing regular OcTree iterator...")

    tree = pyoctomap.OcTree(0.1)

    coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    for coord in coords:
        tree.updateNode(coord, True)

    # Test iterator
    leaf_count = 0
    for leaf in tree.begin_leafs():
        leaf_count += 1
        # Regular OcTree should return defaults
        assert leaf.getColor() == (255, 255, 255), "Regular OcTree should return white color"
        assert leaf.getTimestamp() == 0, "Regular OcTree should return 0 timestamp"

    assert leaf_count == len(coords), f"Expected {len(coords)} leaves, got {leaf_count}"

    print("    ✅ Regular OcTree defaults working")
    return tree


def test_memory_cleanup():
    """Test that iterators are properly cleaned up"""
    print("🧹 Testing memory cleanup...")

    tree = pyoctomap.ColorOcTree(0.1)
    tree.updateNode([1.0, 2.0, 3.0], True)
    tree.setNodeColor([1.0, 2.0, 3.0], 255, 0, 0)

    # Create iterator and weak reference
    iterator = tree.begin_leafs()
    iterator_ref = weakref.ref(iterator)

    # Use iterator to ensure it's fully initialized
    count = 0
    for leaf in iterator:
        count += 1

    assert count == 1, f"Expected 1 leaf, got {count}"

    # Delete iterator and force garbage collection
    del iterator
    gc.collect()

    # Iterator should be garbage collected
    if iterator_ref() is not None:
        print("    ⚠️  Warning: Iterator not immediately garbage collected")
        # Try one more time
        gc.collect()
        if iterator_ref() is not None:
            print("    ❌ Iterator memory leak detected!")
            return False

    print("    ✅ Memory cleanup working")
    return True


def test_exception_safety():
    """Test that iterators handle exceptions safely"""
    print("🛡️  Testing exception safety...")

    # Test with None tree
    iterator = pyoctomap.SimpleLeafIterator(None)
    leaf_count = 0
    for leaf in iterator:
        leaf_count += 1
    assert leaf_count == 0, "Iterator with None tree should be empty"

    print("    ✅ Exception safety working")
    return True


def test_bounding_box_iterator():
    """Test bounding box iterator functionality"""
    print("📦 Testing bounding box iterator...")

    tree = pyoctomap.ColorOcTree(0.1)

    # Create nodes in and out of bounding box
    nodes_data = [
        ([0.5, 0.5, 0.5], (255, 0, 0)),     # Inside BBX
        ([1.5, 1.5, 1.5], (0, 255, 0)),     # Inside BBX
        ([3.0, 3.0, 3.0], (0, 0, 255)),     # Outside BBX
    ]

    for coord, color in nodes_data:
        tree.updateNode(coord, True)
        tree.setNodeColor(coord, color[0], color[1], color[2])

    # Define bounding box
    bbx_min = np.array([0.0, 0.0, 0.0])
    bbx_max = np.array([2.0, 2.0, 2.0])

    # Test BBX iterator
    found_colors = []
    for leaf in tree.begin_leafs_bbx(bbx_min, bbx_max):
        found_colors.append(leaf.getColor())

    # Should find exactly 2 nodes (red and green)
    assert len(found_colors) == 2, f"Expected 2 nodes in BBX, got {len(found_colors)}"
    expected_colors = [(255, 0, 0), (0, 255, 0)]
    for color in found_colors:
        assert color in expected_colors, f"Unexpected color {color} in BBX"

    print("    ✅ Bounding box iterator working")
    return True


def main():
    """Run all iterator tests"""
    print("🧪 PyOctoMap Iterator Functionality Test Suite")
    print("=" * 50)

    tests = [
        ("Color Iterator", test_color_iterator),
        ("Timestamp Iterator", test_timestamp_iterator),
        ("Regular OcTree Iterator", test_regular_octree_iterator),
        ("Memory Cleanup", test_memory_cleanup),
        ("Exception Safety", test_exception_safety),
        ("Bounding Box Iterator", test_bounding_box_iterator),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result is not False:  # Some tests return objects, some return bool
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
            import traceback
            traceback.print_exc()

    print("=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All iterator tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())




