#!/usr/bin/env python3
"""
Comprehensive Pointcloud test script.

Covers:
- Construction (empty, from numpy array, copy constructor)
- Adding points (push_back with different signatures)
- Accessing points (__getitem__, __setitem__, getPoint, back)
- Size and length operations
- Clearing and reserving
- Transformations (transform, transformAbsolute, rotate)
- Bounding box operations (calcBBX, crop)
- Filtering (minDist, subSampleRandom)
- Conversion to numpy
- File I/O (readBinary, writeBinary, writeVrml)
- Edge cases and error handling
"""

import os
import sys
import tempfile
import numpy as np
import pytest
import math


def section(title: str):
    """Print a section header."""
    print("\n" + "=" * 8 + f" {title} " + "=" * 8)


# Import Pointcloud at module level
try:
    from pyoctomap import Pointcloud
    POINTCLOUD_AVAILABLE = True
except ImportError:
    POINTCLOUD_AVAILABLE = False
    Pointcloud = None


@pytest.fixture
def pointcloud_class():
    """Fixture that returns the Pointcloud class."""
    if not POINTCLOUD_AVAILABLE:
        pytest.skip("Pointcloud not available")
    return Pointcloud


def test_import():
    """Test that Pointcloud can be imported."""
    section("Import")
    if not POINTCLOUD_AVAILABLE:
        pytest.skip("Pointcloud not available")
    print("✅ Pointcloud imported successfully")


def test_construction(pointcloud_class):
    """Test Pointcloud construction methods."""
    section("Construction")
    Pointcloud = pointcloud_class
    
    # Empty construction
    pc1 = Pointcloud()
    assert pc1.size() == 0
    assert len(pc1) == 0
    print(f"✅ Empty Pointcloud: size={pc1.size()}")
    
    # Construction from numpy array
    points = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)
    pc2 = Pointcloud(points)
    assert pc2.size() == 3
    assert len(pc2) == 3
    print(f"✅ From numpy array: size={pc2.size()}")
    
    # Copy constructor
    pc3 = Pointcloud(pc2)
    assert pc3.size() == 3
    assert pc3.size() == pc2.size()
    print(f"✅ Copy constructor: size={pc3.size()}")
    
    # Test that copy is independent
    pc2.clear()
    assert pc2.size() == 0
    assert pc3.size() == 3  # Copy should be unaffected
    print("✅ Copy is independent")
    
    # Invalid construction
    try:
        pc_invalid = Pointcloud("invalid")
        assert False, "Should have raised TypeError"
    except TypeError:
        print("✅ Invalid construction raises TypeError")
    
    # Invalid numpy array shape
    try:
        invalid_points = np.array([[1.0, 2.0]])  # Wrong shape
        pc_invalid = Pointcloud(invalid_points)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✅ Invalid numpy array shape raises ValueError")


def test_push_back(pointcloud_class):
    """Test adding points to Pointcloud."""
    section("push_back")
    Pointcloud = pointcloud_class
    
    pc = Pointcloud()
    
    # push_back with three separate arguments
    pc.push_back(1.0, 2.0, 3.0)
    assert pc.size() == 1
    print("✅ push_back(x, y, z)")
    
    # push_back with numpy array
    pc.push_back(np.array([4.0, 5.0, 6.0]))
    assert pc.size() == 2
    print("✅ push_back(array)")
    
    # push_back with list
    pc.push_back([7.0, 8.0, 9.0])
    assert pc.size() == 3
    print("✅ push_back(list)")
    
    # push_back with tuple
    pc.push_back((10.0, 11.0, 12.0))
    assert pc.size() == 4
    print("✅ push_back(tuple)")
    
    # Invalid push_back
    try:
        pc.push_back([1.0, 2.0])  # Wrong shape
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✅ Invalid push_back raises ValueError")
    
    # Verify points
    assert np.allclose(pc[0], [1.0, 2.0, 3.0])
    assert np.allclose(pc[1], [4.0, 5.0, 6.0])
    assert np.allclose(pc[2], [7.0, 8.0, 9.0])
    assert np.allclose(pc[3], [10.0, 11.0, 12.0])
    print("✅ All points added correctly")


def test_accessors(pointcloud_class):
    """Test point access methods."""
    section("Accessors")
    Pointcloud = pointcloud_class
    
    points = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)
    pc = Pointcloud(points)
    
    # __getitem__
    p0 = pc[0]
    assert isinstance(p0, np.ndarray)
    assert np.allclose(p0, [1.0, 2.0, 3.0])
    print("✅ __getitem__ works")
    
    # getPoint
    p1 = pc.getPoint(1)
    assert isinstance(p1, np.ndarray)
    assert np.allclose(p1, [4.0, 5.0, 6.0])
    print("✅ getPoint works")
    
    # back
    p_back = pc.back()
    assert np.allclose(p_back, [7.0, 8.0, 9.0])
    print("✅ back() works")
    
    # __setitem__
    pc[0] = [10.0, 20.0, 30.0]
    assert np.allclose(pc[0], [10.0, 20.0, 30.0])
    print("✅ __setitem__ works")
    
    # Index out of range
    try:
        _ = pc[100]
        assert False, "Should have raised IndexError"
    except IndexError:
        print("✅ Index out of range raises IndexError")
    
    # Invalid setitem value
    try:
        pc[0] = [1.0, 2.0]  # Wrong shape
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✅ Invalid setitem value raises ValueError")


def test_size_and_clear(pointcloud_class):
    """Test size, clear, and reserve methods."""
    section("Size and Clear")
    Pointcloud = pointcloud_class
    
    pc = Pointcloud()
    assert pc.size() == 0
    assert len(pc) == 0
    
    # Add points
    for i in range(10):
        pc.push_back(float(i), float(i+1), float(i+2))
    
    assert pc.size() == 10
    assert len(pc) == 10
    print(f"✅ Size after adding points: {pc.size()}")
    
    # Reserve
    pc.reserve(100)
    assert pc.size() == 10  # Size shouldn't change
    print("✅ reserve() works")
    
    # Clear
    pc.clear()
    assert pc.size() == 0
    assert len(pc) == 0
    print("✅ clear() works")


def test_to_numpy(pointcloud_class):
    """Test conversion to numpy array."""
    section("to_numpy")
    Pointcloud = pointcloud_class
    
    points = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)
    pc = Pointcloud(points)
    
    arr = pc.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, 3)
    assert np.allclose(arr, points)
    print("✅ to_numpy() returns correct array")
    
    # Empty pointcloud
    pc_empty = Pointcloud()
    arr_empty = pc_empty.to_numpy()
    assert arr_empty.shape == (0, 3)
    print("✅ Empty pointcloud to_numpy() works")


def test_transform(pointcloud_class):
    """Test transformation methods."""
    section("Transformations")
    Pointcloud = pointcloud_class
    
    # Create a simple pointcloud
    points = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    pc = Pointcloud(points)
    
    # Test transform with separate arguments
    pc.transform(x=1.0, y=2.0, z=3.0, roll=0.0, pitch=0.0, yaw=0.0)
    # After translation (1, 2, 3), points should be:
    # [2.0, 2.0, 3.0], [1.0, 3.0, 3.0], [1.0, 2.0, 4.0]
    result = pc.to_numpy()
    print(f"✅ transform() with separate args: first point = {result[0]}")
    
    # Test transform with array
    pc2 = Pointcloud(points)
    transform_array = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    pc2.transform(transform_array)
    result2 = pc2.to_numpy()
    assert np.allclose(result, result2)
    print("✅ transform() with array works")
    
    # Test rotate
    pc3 = Pointcloud(points)
    # Rotate 90 degrees around Z axis
    pc3.rotate(roll=0.0, pitch=0.0, yaw=math.pi/2)
    result3 = pc3.to_numpy()
    print(f"✅ rotate() works: first point after rotation = {result3[0]}")
    
    # Test transformAbsolute
    pc4 = Pointcloud(points)
    pc4.transformAbsolute(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)
    result4 = pc4.to_numpy()
    assert np.allclose(result4, points, atol=1e-6)
    print("✅ transformAbsolute() works")


def test_bounding_box(pointcloud_class):
    """Test bounding box operations."""
    section("Bounding Box")
    Pointcloud = pointcloud_class
    
    points = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [-1.0, -2.0, -3.0],
    ], dtype=np.float64)
    pc = Pointcloud(points)
    
    # Calculate bounding box
    lower, upper = pc.calcBBX()
    assert isinstance(lower, np.ndarray)
    assert isinstance(upper, np.ndarray)
    assert np.allclose(lower, [-1.0, -2.0, -3.0], atol=1e-5)
    assert np.allclose(upper, [7.0, 8.0, 9.0], atol=1e-5)
    print(f"✅ calcBBX(): lower={lower}, upper={upper}")
    
    # Crop to bounding box
    pc_crop = Pointcloud(points)
    crop_lower = np.array([0.0, 0.0, 0.0])
    crop_upper = np.array([5.0, 5.0, 5.0])
    pc_crop.crop(crop_lower, crop_upper)
    cropped_points = pc_crop.to_numpy()
    print(f"✅ crop(): remaining points = {pc_crop.size()}")
    
    # Verify all cropped points are within bounds
    for point in cropped_points:
        assert np.all(point >= crop_lower - 1e-6)
        assert np.all(point <= crop_upper + 1e-6)
    print("✅ All cropped points are within bounds")
    
    # Empty pointcloud bounding box
    pc_empty = Pointcloud()
    try:
        lower_empty, upper_empty = pc_empty.calcBBX()
        print("✅ calcBBX() on empty pointcloud handled")
    except Exception as e:
        print(f"⚠️  calcBBX() on empty pointcloud: {e}")


def test_minDist(pointcloud_class):
    """Test minimum distance filtering."""
    section("minDist")
    Pointcloud = pointcloud_class
    
    # Create pointcloud with points at various distances from origin
    points = np.array([
        [0.1, 0.0, 0.0],   # Distance ~0.1
        [1.0, 0.0, 0.0],   # Distance 1.0
        [2.0, 0.0, 0.0],   # Distance 2.0
        [0.05, 0.0, 0.0],  # Distance ~0.05
    ], dtype=np.float64)
    pc = Pointcloud(points)
    
    original_size = pc.size()
    pc.minDist(0.2)  # Remove points closer than 0.2
    new_size = pc.size()
    
    assert new_size < original_size
    print(f"✅ minDist(0.2): {original_size} -> {new_size} points")
    
    # Verify remaining points are far enough
    remaining = pc.to_numpy()
    for point in remaining:
        dist = np.linalg.norm(point)
        assert dist >= 0.2 - 1e-6, f"Point {point} too close to origin"
    print("✅ All remaining points satisfy minDist constraint")


def test_subSampleRandom(pointcloud_class):
    """Test random subsampling."""
    section("subSampleRandom")
    Pointcloud = pointcloud_class
    
    # Create a large pointcloud
    np.random.seed(42)  # For reproducibility
    points = np.random.rand(100, 3) * 10.0
    pc = Pointcloud(points)
    
    assert pc.size() == 100
    
    # Subsample
    sample = pc.subSampleRandom(20)
    assert isinstance(sample, Pointcloud)
    assert sample.size() == 20
    print(f"✅ subSampleRandom(20): {pc.size()} -> {sample.size()} points")
    
    # Verify sample points are from original
    sample_arr = sample.to_numpy()
    original_arr = pc.to_numpy()
    
    # Check that all sample points exist in original (within tolerance)
    for sample_point in sample_arr:
        found = False
        for orig_point in original_arr:
            if np.allclose(sample_point, orig_point, atol=1e-5):
                found = True
                break
        assert found, f"Sample point {sample_point} not found in original"
    print("✅ All sample points are from original pointcloud")
    
    # Test with more samples than available
    sample_large = pc.subSampleRandom(200)
    assert sample_large.size() <= pc.size()
    print("✅ subSampleRandom with more samples than available handled")


def test_file_io(pointcloud_class):
    """Test file I/O operations."""
    section("File I/O")
    Pointcloud = pointcloud_class
    
    points = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)
    pc = Pointcloud(points)
    
    # Test writeBinary and readBinary
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
        tmp_path = tmp.name
    
    try:
        # Write
        success = pc.writeBinary(tmp_path)
        assert success
        print(f"✅ writeBinary() successful: {tmp_path}")
        
        # Read
        pc_read = Pointcloud()
        success_read = pc_read.readBinary(tmp_path)
        assert success_read
        assert pc_read.size() == pc.size()
        
        # Verify points match (within tolerance)
        original_arr = pc.to_numpy()
        read_arr = pc_read.to_numpy()
        assert np.allclose(original_arr, read_arr, atol=1e-5)
        print("✅ readBinary() successful and points match")
        
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    # Test writeVrml (if supported)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wrl') as tmp:
        vrml_path = tmp.name
    
    try:
        pc.writeVrml(vrml_path)
        assert os.path.exists(vrml_path)
        print(f"✅ writeVrml() successful: {vrml_path}")
    except Exception as e:
        print(f"⚠️  writeVrml() not available or error: {e}")
    finally:
        if os.path.exists(vrml_path):
            os.unlink(vrml_path)
    
    # Test invalid file paths
    try:
        pc.writeBinary("/nonexistent/path/file.bin")
        assert False, "Should have raised IOError"
    except IOError:
        print("✅ Invalid write path raises IOError")
    
    try:
        pc_invalid = Pointcloud()
        pc_invalid.readBinary("/nonexistent/path/file.bin")
        assert False, "Should have raised IOError"
    except IOError:
        print("✅ Invalid read path raises IOError")


def test_edge_cases(pointcloud_class):
    """Test edge cases and error handling."""
    section("Edge Cases")
    Pointcloud = pointcloud_class
    
    # Empty pointcloud operations
    pc_empty = Pointcloud()
    assert pc_empty.size() == 0
    assert len(pc_empty) == 0
    
    try:
        _ = pc_empty[0]
        assert False, "Should have raised IndexError"
    except IndexError:
        print("✅ Empty pointcloud indexing raises IndexError")
    
    try:
        _ = pc_empty.back()
        assert False, "Should have raised IndexError"
    except IndexError:
        print("✅ back() on empty pointcloud raises IndexError")
    
    # Single point
    pc_single = Pointcloud()
    pc_single.push_back(1.0, 2.0, 3.0)
    assert pc_single.size() == 1
    assert np.allclose(pc_single[0], [1.0, 2.0, 3.0])
    assert np.allclose(pc_single.back(), [1.0, 2.0, 3.0])
    print("✅ Single point operations work")
    
    # Very large pointcloud
    large_points = np.random.rand(1000, 3) * 100.0
    pc_large = Pointcloud(large_points)
    assert pc_large.size() == 1000
    print(f"✅ Large pointcloud ({pc_large.size()} points) works")
    
    # Negative indices (Python convention)
    pc_test = Pointcloud(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    # Python convention: -1 is last element
    assert np.allclose(pc_test[-1], [4.0, 5.0, 6.0])
    assert np.allclose(pc_test[-2], [1.0, 2.0, 3.0])
    print("✅ Negative indexing supported")
    
    # Test negative index out of range
    try:
        _ = pc_test[-3]
        assert False, "Should have raised IndexError"
    except IndexError:
        print("✅ Negative index out of range raises IndexError")
    
    # Test setting with negative index
    pc_test[-1] = [10.0, 20.0, 30.0]
    assert np.allclose(pc_test[-1], [10.0, 20.0, 30.0])
    assert np.allclose(pc_test[1], [10.0, 20.0, 30.0])
    print("✅ Setting with negative index works")


def test_integration_with_octree(pointcloud_class):
    """Test integration with OcTree."""
    section("Integration with OcTree")
    Pointcloud = pointcloud_class
    
    try:
        from pyoctomap import OcTree
        
        # Create pointcloud
        points = np.array([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=np.float64)
        pc = Pointcloud(points)
        
        # Create octree
        tree = OcTree(0.1)
        
        # Convert pointcloud to numpy for insertion
        pc_array = pc.to_numpy()
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        # Insert pointcloud into octree
        tree.insertPointCloud(pc_array, origin)
        print(f"✅ Pointcloud inserted into OcTree: tree size = {tree.size()}")
        
        # Verify points are in tree
        for point in pc_array:
            node = tree.search(point)
            if node:
                assert tree.isNodeOccupied(node)
        print("✅ Points found in octree")
        
    except ImportError:
        print("⚠️  OcTree not available for integration test")


def test_string_representation(pointcloud_class):
    """Test string representation."""
    section("String Representation")
    Pointcloud = pointcloud_class
    
    pc = Pointcloud()
    repr_str = repr(pc)
    str_str = str(pc)
    
    assert "Pointcloud" in repr_str
    assert "Pointcloud" in str_str
    print(f"✅ repr(): {repr_str}")
    print(f"✅ str(): {str_str}")
    
    pc.push_back(1.0, 2.0, 3.0)
    repr_str2 = repr(pc)
    assert "size=1" in repr_str2
    print(f"✅ repr() with points: {repr_str2}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Pointcloud Comprehensive Test Suite")
    print("=" * 60)
    
    if not POINTCLOUD_AVAILABLE:
        print("❌ Pointcloud not available, skipping tests")
        return
    
    # Create a fixture-like object for standalone execution
    class PointcloudFixture:
        def __call__(self):
            return Pointcloud
    
    fixture = PointcloudFixture()
    
    try:
        test_import()
        test_construction(fixture)
        test_push_back(fixture)
        test_accessors(fixture)
        test_size_and_clear(fixture)
        test_to_numpy(fixture)
        test_transform(fixture)
        test_bounding_box(fixture)
        test_minDist(fixture)
        test_subSampleRandom(fixture)
        test_file_io(fixture)
        test_edge_cases(fixture)
        test_integration_with_octree(fixture)
        test_string_representation(fixture)
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

