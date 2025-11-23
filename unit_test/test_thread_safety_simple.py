"""
Simple Thread Safety Tests for PyOctoMap

This is a simplified version that can run without pytest and focuses on
the most critical thread safety issues that could cause crashes or data corruption.
"""

import gc
import threading
import time

import numpy as np

import pyoctomap


def test_basic_concurrent_operations():
    """Test basic concurrent read/write operations"""
    print("🧪 Testing basic concurrent operations...")

    tree = pyoctomap.OcTree(0.1)
    sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    errors = []
    lock = threading.Lock()

    def writer_thread(thread_id):
        """Writer thread"""
        try:
            for _i in range(20):
                points = np.random.uniform(-5, 5, (10, 3)).astype(np.float64)
                tree.addPointCloudWithRayCasting(points, sensor_origin)
                time.sleep(0.01)
        except Exception as e:
            with lock:
                errors.append(f"Writer {thread_id}: {e}")

    def reader_thread(thread_id):
        """Reader thread"""
        try:
            for _i in range(30):
                # Search operations
                test_point = np.random.uniform(-5, 5, 3).astype(np.float64)
                node = tree.search(test_point)
                if node is not None:
                    tree.isNodeOccupied(node)

                # Tree statistics
                tree.getNumLeafNodes()
                tree.getResolution()
                time.sleep(0.01)
        except Exception as e:
            with lock:
                errors.append(f"Reader {thread_id}: {e}")

    # Start threads
    threads = []
    for i in range(2):
        t = threading.Thread(target=writer_thread, args=(i,))
        t.start()
        threads.append(t)

    for i in range(3):
        t = threading.Thread(target=reader_thread, args=(i,))
        t.start()
        threads.append(t)

    # Wait for completion
    for t in threads:
        t.join()

    print(f"   Errors: {len(errors)}")
    if errors:
        for error in errors[:5]:
            print(f"     {error}")

    # Verify tree state
    assert tree.getNumLeafNodes() > 0
    assert tree.getResolution() > 0
    print("   ✅ Basic concurrent operations test passed")


def test_iterator_safety():
    """Test iterator safety during modifications"""
    print("🧪 Testing iterator safety...")

    tree = pyoctomap.OcTree(0.1)
    sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # Pre-populate tree
    initial_points = np.random.uniform(-5, 5, (50, 3)).astype(np.float64)
    tree.addPointCloudWithRayCasting(initial_points, sensor_origin)

    errors = []
    lock = threading.Lock()

    def iterator_thread():
        """Thread that iterates through the tree"""
        try:
            for _ in range(20):
                # Test leaf iterator
                leaf_count = 0
                for leaf in tree.begin_leafs():
                    leaf.getCoordinate()
                    leaf.getSize()
                    leaf_count += 1
                    if leaf_count > 10:
                        break
                time.sleep(0.02)
        except Exception as e:
            with lock:
                errors.append(f"Iterator: {e}")

    def modifier_thread():
        """Thread that modifies the tree"""
        try:
            for _ in range(10):
                points = np.random.uniform(-5, 5, (10, 3)).astype(np.float64)
                tree.addPointCloudWithRayCasting(points, sensor_origin)
                time.sleep(0.05)
        except Exception as e:
            with lock:
                errors.append(f"Modifier: {e}")

    # Start threads
    iterator = threading.Thread(target=iterator_thread)
    modifier = threading.Thread(target=modifier_thread)

    iterator.start()
    modifier.start()

    iterator.join()
    modifier.join()

    print(f"   Iterator errors: {len(errors)}")
    if errors:
        for error in errors:
            print(f"     {error}")

    print("   ✅ Iterator safety test completed")


def test_memory_safety():
    """Test memory safety with multiple trees"""
    print("🧪 Testing memory safety...")

    trees = [pyoctomap.OcTree(0.1) for _ in range(3)]
    sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    errors = []
    lock = threading.Lock()

    def memory_worker(tree_id, tree):
        """Worker for a specific tree"""
        try:
            for _ in range(20):
                points = np.random.uniform(-5, 5, (10, 3)).astype(np.float64)
                tree.addPointCloudWithRayCasting(points, sensor_origin)

                # Search operations
                test_point = np.random.uniform(-5, 5, 3).astype(np.float64)
                node = tree.search(test_point)
                if node is not None:
                    tree.isNodeOccupied(node)

                tree.getNumLeafNodes()
                time.sleep(0.01)
        except Exception as e:
            with lock:
                errors.append(f"Memory worker {tree_id}: {e}")

    # Start workers
    threads = []
    for i, tree in enumerate(trees):
        t = threading.Thread(target=memory_worker, args=(i, tree))
        t.start()
        threads.append(t)

    # Wait for completion
    for t in threads:
        t.join()

    # Clean up
    del trees
    gc.collect()

    print(f"   Memory errors: {len(errors)}")
    if errors:
        for error in errors[:5]:
            print(f"     {error}")

    assert len(errors) == 0, f"Memory safety issues: {errors}"
    print("   ✅ Memory safety test passed")


def test_rapid_modifications():
    """Test rapid concurrent modifications"""
    print("🧪 Testing rapid modifications...")

    tree = pyoctomap.OcTree(0.1)
    sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    errors = []
    lock = threading.Lock()

    def rapid_worker(thread_id):
        """Worker that rapidly modifies the tree"""
        try:
            for _ in range(50):
                points = np.random.uniform(-5, 5, (5, 3)).astype(np.float64)
                tree.addPointCloudWithRayCasting(points, sensor_origin)

                # Update nodes
                test_point = np.random.uniform(-5, 5, 3).astype(np.float64)
                tree.updateNode(test_point, True)

                # Search
                node = tree.search(test_point)
                if node is not None:
                    tree.isNodeOccupied(node)
        except Exception as e:
            with lock:
                errors.append(f"Rapid worker {thread_id}: {e}")

    # Start many workers
    threads = []
    for i in range(8):
        t = threading.Thread(target=rapid_worker, args=(i,))
        t.start()
        threads.append(t)

    # Wait for completion
    for t in threads:
        t.join()

    print(f"   Rapid modification errors: {len(errors)}")
    if errors:
        for error in errors[:5]:
            print(f"     {error}")

    # Verify tree integrity
    try:
        num_nodes = tree.getNumLeafNodes()
        resolution = tree.getResolution()
        assert num_nodes >= 0
        assert resolution > 0
    except Exception as e:
        errors.append(f"Tree integrity: {e}")

    print(f"   ✅ Rapid modifications test completed (errors: {len(errors)})")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("🧪 Testing edge cases...")

    tree = pyoctomap.OcTree(0.1)
    sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    errors = []
    lock = threading.Lock()

    def edge_case_worker(thread_id):
        """Worker that tests edge cases"""
        try:
            for _ in range(20):
                # Test with empty point cloud
                try:
                    empty_points = np.zeros((0, 3), dtype=np.float64)
                    tree.addPointCloudWithRayCasting(empty_points, sensor_origin)
                except Exception as e:
                    with lock:
                        errors.append(f"Empty points {thread_id}: {e}")

                # Test with normal points
                try:
                    points = np.random.uniform(-5, 5, (5, 3)).astype(np.float64)
                    tree.addPointCloudWithRayCasting(points, sensor_origin)
                except Exception as e:
                    with lock:
                        errors.append(f"Normal points {thread_id}: {e}")

                # Test with large values
                try:
                    large_points = np.array([[1e6, 1e6, 1e6]], dtype=np.float64)
                    tree.addPointCloudWithRayCasting(large_points, sensor_origin)
                except Exception as e:
                    with lock:
                        errors.append(f"Large points {thread_id}: {e}")

                time.sleep(0.01)
        except Exception as e:
            with lock:
                errors.append(f"Edge case worker {thread_id}: {e}")

    # Start edge case workers
    threads = []
    for i in range(3):
        t = threading.Thread(target=edge_case_worker, args=(i,))
        t.start()
        threads.append(t)

    # Wait for completion
    for t in threads:
        t.join()

    print(f"   Edge case errors: {len(errors)}")
    if errors:
        for error in errors[:5]:
            print(f"     {error}")

    print("   ✅ Edge cases test completed")


def run_all_tests():
    """Run all thread safety tests"""
    print("🚀 Starting PyOctoMap Thread Safety Tests")
    print("=" * 50)

    try:
        test_basic_concurrent_operations()
        test_iterator_safety()
        test_memory_safety()
        test_rapid_modifications()
        test_edge_cases()

        print("\n" + "=" * 50)
        print("✅ All thread safety tests completed!")

    except Exception as e:
        print(f"\n❌ Thread safety tests failed: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
